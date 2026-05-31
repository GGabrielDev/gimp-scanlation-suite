import os
import sys
import json
import base64
import io
import numpy as np
from PIL import Image

from server.core.config import MODELS_CONFIG
from server.services.model_loader import get_or_load_model, unload_model

def generate_vlm_prompt(api_base: str, default_prompt: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return default_prompt or "manga reconstruction, detailed background, high quality line art"
        
    try:
        import requests
        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        user_msg = (
            "You are an AI assistant that writes prompts for Stable Diffusion inpainting in manga. "
            "We need to erase a text bubble. Write a short, comma-separated list of prompt keywords "
            "describing the underlying manga artwork (e.g., screentone, lines, character hair/clothes) "
            "that should fill the space. Keep it under 20 words. Output ONLY the comma-separated keywords."
        )
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.5,
            "max_tokens": 50
        }
        response = requests.post(url, json=payload, headers=headers, timeout=10.0)
        response.raise_for_status()
        res_json = response.json()
        gpt_prompt = res_json["choices"][0]["message"]["content"].strip()
        gpt_prompt = gpt_prompt.strip('"\'')
        return gpt_prompt
    except Exception as e:
        sys.stderr.write(f"[Server Inpaint Service] Failed to generate auto-prompt via VLM: {e}\n")
        return default_prompt or "manga reconstruction, detailed background, high quality line art"

def run_inpaint_generator(model: str, batch_payload: list, options: dict):
    """
    Generator yielding newline-separated JSON progress strings, ending with inpainted base64.
    """
    try:
        yield json.dumps({"type": "progress", "percentage": 0.0, "message": "Loading inpainting model..."}) + "\n"
        session = get_or_load_model(model)
        
        yield json.dumps({"type": "progress", "percentage": 0.3, "message": "Decoding image and mask..."}) + "\n"
        
        img_str = batch_payload[0]
        mask_str = batch_payload[1]
        
        header1, data1 = img_str.split(",", 1) if "," in img_str else ("", img_str)
        header2, data2 = mask_str.split(",", 1) if "," in mask_str else ("", mask_str)
        
        pil_img = Image.open(io.BytesIO(base64.b64decode(data1))).convert("RGB")
        pil_mask = Image.open(io.BytesIO(base64.b64decode(data2))).convert("L")
        
        img_np = np.array(pil_img)
        mask_np = np.array(pil_mask)
        
        full_h, full_w = img_np.shape[:2]
        
        # Get bounding boxes from options (passed by client)
        bounding_boxes = options.get("bounding_boxes", [])

        if not bounding_boxes:
            # Fallback to single bounding box covering all white pixels of mask
            y_indices, x_indices = np.where(mask_np > 0)
            if len(x_indices) > 0:
                bounding_boxes = [(int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max()))]

        out_img_np = img_np.copy()
        
        is_diffusion = MODELS_CONFIG.get(model, {}).get("handler_class") == "DiffusionInpainting"

        if bounding_boxes:
            yield json.dumps({"type": "progress", "percentage": 0.6, "message": f"Running crop-based inpainting on {len(bounding_boxes)} regions..."}) + "\n"
            
            # Setup SD parameters if Diffusion
            if is_diffusion:
                prompt = options.get("prompt") or ""
                negative_prompt = options.get("negative_prompt") or "color, colorful, low quality, blurry, bad anatomy"
                steps = int(options.get("steps") or 25)
                guidance_scale = float(options.get("guidance_scale") or 7.5)
                auto_prompt = bool(options.get("auto_prompt") or False)
                consensus_arbiter = options.get("consensus_arbiter") or "DeepSeek"
                
                # If auto-prompt is enabled and prompt is empty, generate it
                if auto_prompt and not prompt:
                    yield json.dumps({"type": "progress", "percentage": 0.5, "message": "Generating prompt via VLM..."}) + "\n"
                    api_base = os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
                    prompt = generate_vlm_prompt(api_base, prompt)
                    sys.stderr.write(f"[Server Inpaint Service] VLM Auto-Prompt: '{prompt}'\n")
                elif not prompt:
                    prompt = "manga reconstruction, detailed background, high quality line art"

            for idx, box in enumerate(bounding_boxes):
                xmin, ymin, xmax, ymax = box
                w = xmax - xmin
                h = ymax - ymin
                if w <= 0 or h <= 0:
                    continue
                    
                # Centered square with a margin of 1.6x max dimension
                side = int(max(w, h) * 1.6)
                if side < 256:
                    side = 256
                
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2
                
                x0 = int(cx - side // 2)
                x1 = x0 + side
                y0 = int(cy - side // 2)
                y1 = y0 + side
                
                x0_clipped = max(0, x0)
                x1_clipped = min(full_w, x1)
                y0_clipped = max(0, y0)
                y1_clipped = min(full_h, y1)
                
                crop_w = x1_clipped - x0_clipped
                crop_h = y1_clipped - y0_clipped
                if crop_w <= 0 or crop_h <= 0:
                    continue
                    
                crop_img = img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                crop_mask = mask_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                
                # Resize to 512x512
                crop_img_pil = Image.fromarray(crop_img).resize((512, 512), Image.Resampling.BILINEAR)
                crop_mask_pil = Image.fromarray(crop_mask).resize((512, 512), Image.Resampling.NEAREST)
                
                if is_diffusion:
                    import torch
                    # Create generator for reproducibility
                    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                    
                    # Run SD pipeline
                    res_pil = session(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=crop_img_pil,
                        mask_image=crop_mask_pil,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]
                    
                    # Resize back
                    out_crop_pil = res_pil.resize((crop_w, crop_h), Image.Resampling.BILINEAR)
                    out_crop_original = np.array(out_crop_pil)
                else:
                    crop_img_512 = np.array(crop_img_pil)
                    crop_mask_512 = np.array(crop_mask_pil)
                    
                    img_feed = crop_img_512.astype(np.float32) / 255.0
                    img_feed = np.transpose(img_feed, (2, 0, 1))
                    img_feed = np.expand_dims(img_feed, axis=0)
                    
                    mask_feed = crop_mask_512.astype(np.float32) / 255.0
                    mask_feed = np.expand_dims(mask_feed, axis=0)
                    mask_feed = np.expand_dims(mask_feed, axis=0)
                    
                    # Zero out the masked region in the input image for correct in-distribution inference
                    img_feed = img_feed * (1.0 - mask_feed)
                    
                    input_names = [i.name for i in session.get_inputs()]
                    feeds = {}
                    for name in input_names:
                        if "image" in name.lower() or "input" in name.lower():
                            feeds[name] = img_feed
                        elif "mask" in name.lower():
                            feeds[name] = mask_feed
                    if len(feeds) < 2:
                        feeds = {input_names[0]: img_feed, input_names[1]: mask_feed}
                        
                    outputs = session.run(None, feeds)
                    out_crop = outputs[0]
                    
                    out_crop = np.squeeze(out_crop, axis=0)
                    out_crop = np.transpose(out_crop, (1, 2, 0))
                    out_crop = np.clip(out_crop * 255.0, 0.0, 255.0).astype(np.uint8)
                    
                    # Resize back
                    out_crop_pil = Image.fromarray(out_crop).resize((crop_w, crop_h), Image.Resampling.BILINEAR)
                    out_crop_original = np.array(out_crop_pil)
                
                # Blend using the original crop mask
                mask_area = (crop_mask > 0)[:, :, np.newaxis]
                out_img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped] = np.where(
                    mask_area,
                    out_crop_original,
                    out_img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                )
        
        final_img = out_img_np
        
        yield json.dumps({"type": "progress", "percentage": 0.8, "message": "Encoding result..."}) + "\n"
        out_pil = Image.fromarray(final_img)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": "Done."}) + "\n"
        yield json.dumps({"type": "result", "results": [out_b64]}) + "\n"
    except Exception as e:
        sys.stderr.write(f"[Server Inpaint Service] Inpaint error: {e}\n")
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": f"Error: {e}"}) + "\n"
        yield json.dumps({"type": "result", "results": []}) + "\n"
    finally:
        unload_model(model)
