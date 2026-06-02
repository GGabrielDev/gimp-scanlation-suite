import os
import sys
import json
import base64
import io
import numpy as np
import time
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

def generate_local_vlm_prompt(vlm, crop_img_pil: Image.Image) -> str:
    """
    Queries a local VLM model object with the crop image to generate keywords for Stable Diffusion inpainting.
    """
    try:
        import io
        import base64
        
        # Convert PIL image to base64 data URL
        buf = io.BytesIO()
        crop_img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_url = f"data:image/png;base64,{img_b64}"
        
        vlm.reset()
        prompt_text = (
            "Describe the drawing in this manga crop (screentone, lines, background pattern, hair, clothes). "
            "Ignore the text and bubble, focus on the artwork. "
            "Output ONLY a short list of comma-separated keywords, under 10 words."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        response = vlm.create_chat_completion(
            messages=messages,
            max_tokens=30,
            temperature=0.2
        )
        res_text = response["choices"][0]["message"]["content"].strip()
        res_text = res_text.strip('"\'')
        if ":" in res_text and not "," in res_text.split(":", 1)[0]:
            res_text = res_text.split(":", 1)[1].strip()
        
        # Clean and limit to at most 12 keywords
        parts = [p.strip() for p in res_text.split(",") if p.strip()]
        parts = parts[:12]
        res_text = ", ".join(parts)
        return res_text
    except Exception as e:
        sys.stderr.write(f"[Server Inpaint Service] Local VLM auto-prompt generation failed: {e}\n")
        return "manga reconstruction, detailed background, high quality line art"

def calculate_dynamic_crop_side(w: int, h: int) -> int:
    """
    Dynamically determines the side length of the square inpainting crop.
    - Small bubbles get a fixed 256px context window to ensure enough surrounding context.
    - Medium bubbles smoothly scale the expansion multiplier from 1.5x down to 1.25x.
    - Large bubbles scale down to 1.1x to prevent severe resolution downscaling when resized to 512x512.
    """
    base = max(w, h)
    if base <= 160:
        return 256
    elif base <= 256:
        # Interpolate crop size from 256px to 384px
        return int(256 + (base - 160) * (384 - 256) / (256 - 160))
    elif base <= 768:
        # Interpolate multiplier from 1.5x down to 1.25x
        mult = 1.5 - (base - 256) / (768 - 256) * 0.25
        return int(base * mult)
    else:
        # Scale down to minimum 1.1x multiplier for extremely large regions
        mult = max(1.1, 1.25 - (base - 768) / 1000 * 0.15)
        return int(base * mult)

def run_inpaint_generator(model: str, batch_payload: list, options: dict):

    """
    Generator yielding newline-separated JSON progress strings, ending with inpainted base64.
    """
    total_start = time.time()
    try:
        yield json.dumps({"type": "progress", "percentage": 0.0, "message": "Loading inpainting model..."}) + "\n"
        
        is_diffusion = model in MODELS_CONFIG and MODELS_CONFIG[model].handler_class == "DiffusionInpainting"
        if not is_diffusion:
            session = get_or_load_model(model)
        else:
            session = None
        
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

        sys.stderr.write(f"[Server Inpaint Service] Decoded image ({full_w}x{full_h}) and mask. Found {len(bounding_boxes)} regions.\n")


        if not bounding_boxes:
            # Fallback to single bounding box covering all white pixels of mask
            y_indices, x_indices = np.where(mask_np > 0)
            if len(x_indices) > 0:
                bounding_boxes = [(int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max()))]

        out_img_np = img_np.copy()
        
        if bounding_boxes:
            # Setup SD parameters if Diffusion
            if is_diffusion:
                prompt_input = options.get("prompt") or ""
                negative_prompt = options.get("negative_prompt") or "color, colorful, low quality, blurry, bad anatomy"
                steps = int(options.get("steps") or 25)
                guidance_scale = float(options.get("guidance_scale") or 7.5)
                auto_prompt = bool(options.get("auto_prompt") or False)
                consensus_arbiter = options.get("consensus_arbiter") or "DeepSeek"
                
                prompts = []
                is_local_vlm = auto_prompt and consensus_arbiter in ["olmOCR2_Q4", "olmOCR2_Q6", "olmOCR2_Q8", "PaddleOCR_Manga"]
                
                if auto_prompt and not prompt_input:
                    if is_local_vlm:
                        yield json.dumps({"type": "progress", "percentage": 0.4, "message": f"Generating prompts using local VLM {consensus_arbiter}..."}) + "\n"
                        vlm = get_or_load_model(consensus_arbiter)
                        if vlm:
                            for idx, box in enumerate(bounding_boxes):
                                xmin, ymin, xmax, ymax = box
                                w = xmax - xmin
                                h = ymax - ymin
                                side = calculate_dynamic_crop_side(w, h)
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
                                    prompts.append("manga reconstruction, detailed background, high quality line art")
                                    continue
                                
                                crop_img = img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                                crop_img_pil = Image.fromarray(crop_img).resize((512, 512), Image.Resampling.BILINEAR)
                                
                                vlm_prompt = generate_local_vlm_prompt(vlm, crop_img_pil)
                                sys.stderr.write(f"[Server Inpaint Service] VLM Local Auto-Prompt for region {idx}: '{vlm_prompt}'\n")
                                prompts.append(vlm_prompt)
                            unload_model(consensus_arbiter)
                        else:
                            prompts = ["manga reconstruction, detailed background, high quality line art"] * len(bounding_boxes)
                    else:
                        yield json.dumps({"type": "progress", "percentage": 0.4, "message": "Generating prompt via VLM API..."}) + "\n"
                        api_base = os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
                        api_prompt = generate_vlm_prompt(api_base, prompt_input)
                        sys.stderr.write(f"[Server Inpaint Service] VLM API Auto-Prompt: '{api_prompt}'\n")
                        prompts = [api_prompt] * len(bounding_boxes)
                else:
                    general_prompt = prompt_input or "manga reconstruction, detailed background, high quality line art"
                    prompts = [general_prompt] * len(bounding_boxes)
                
                # Now load the diffusion inpainting model
                yield json.dumps({"type": "progress", "percentage": 0.55, "message": "Loading diffusion inpainting model..."}) + "\n"
                session = get_or_load_model(model)
            
            yield json.dumps({"type": "progress", "percentage": 0.6, "message": f"Running crop-based inpainting on {len(bounding_boxes)} regions..."}) + "\n"

            for idx, box in enumerate(bounding_boxes):
                xmin, ymin, xmax, ymax = box
                w = xmax - xmin
                h = ymax - ymin
                if w <= 0 or h <= 0:
                    sys.stderr.write(f"[Server Inpaint Service] Region {idx+1}/{len(bounding_boxes)}: Skipped due to invalid dimensions ({w}x{h})\n")
                    continue
                    
                # Centered square with a dynamic context window
                side = calculate_dynamic_crop_side(w, h)
                
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
                    sys.stderr.write(f"[Server Inpaint Service] Region {idx+1}/{len(bounding_boxes)}: Skipped (Clipped size <= 0)\n")
                    continue
                    
                sys.stderr.write(f"[Server Inpaint Service] Region {idx+1}/{len(bounding_boxes)}: Original Box=[{xmin}, {ymin}, {xmax}, {ymax}] (Size={w}x{h}) | Dynamic Side={side}px | Crop Box=[{x0_clipped}, {y0_clipped}, {x1_clipped}, {y1_clipped}] (Actual Size={crop_w}x{crop_h})\n")
                
                crop_img = img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]

                crop_mask = mask_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                
                # Resize to 512x512
                crop_img_pil = Image.fromarray(crop_img).resize((512, 512), Image.Resampling.BILINEAR)
                crop_mask_pil = Image.fromarray(crop_mask).resize((512, 512), Image.Resampling.NEAREST)
                
                inf_start = time.time()
                
                if is_diffusion:
                    import torch

                    # Create generator for reproducibility
                    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                    
                    # Run SD pipeline with individual prompt
                    prompt = prompts[idx] if idx < len(prompts) else "manga reconstruction, detailed background, high quality line art"
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
                
                elapsed = time.time() - inf_start
                sys.stderr.write(f"[Server Inpaint Service] Region {idx+1}/{len(bounding_boxes)}: Inference completed in {elapsed:.2f} seconds\n")
                
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
        
        total_elapsed = time.time() - total_start
        sys.stderr.write(f"[Server Inpaint Service] Completed inpainting of {len(bounding_boxes)} regions in {total_elapsed:.2f} seconds (avg {total_elapsed/len(bounding_boxes) if bounding_boxes else 0:.2f}s per region).\n")
        
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": "Done."}) + "\n"
        yield json.dumps({"type": "result", "results": [out_b64]}) + "\n"
    except Exception as e:
        sys.stderr.write(f"[Server Inpaint Service] Inpaint error: {e}\n")
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": f"Error: {e}"}) + "\n"
        yield json.dumps({"type": "result", "results": []}) + "\n"
    finally:
        unload_model(model)
