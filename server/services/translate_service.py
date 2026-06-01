import os
import sys
import json
import requests
import time

from server.core.config import MODELS_CONFIG
from server.services.model_loader import get_or_load_model, unload_model

def run_translate_generator(model: str, batch_payload: list, options: dict):
    """
    Generator yielding newline-separated JSON progress strings, ending with translate result payload.
    """
    start_time = time.time()
    try:
        yield json.dumps({"type": "progress", "percentage": 0.0, "message": "Loading translation model..."}) + "\n"
        model_cfg = MODELS_CONFIG[model]
        handler_class = model_cfg.get("handler_class")
        
        # Fetch translation options
        src_lang = options.get("source_language") or "Japanese"
        # Since client sends options keys as target_language/source_language (which gets translated to Options in API payload)
        # let's support target_language and target-language keys
        tgt_lang = options.get("target_language") or options.get("target-language") or "English"
        global_ctx = options.get("global_context") or options.get("global-context") or ""
        enable_thinking = options.get("enable_thinking") or options.get("enable-thinking") or False
        
        # Payload is a list of dialogue blocks
        dialogues = batch_payload
        
        # Calculate character and block statistics
        block_count = len(dialogues)
        total_chars = sum(len(str(item.get("text", ""))) for item in dialogues)
        sys.stderr.write(
            f"[Server Translate Service] Received translation request | Model: {model} | "
            f"Source Lang: {src_lang} -> Target Lang: {tgt_lang} | "
            f"Blocks: {block_count} | Total Input Chars: {total_chars}\n"
        )
        
        # Construct the prompt
        system_prompt = (
            f"You are a professional manga/comic translation assistant. "
            f"Translate the following {src_lang} dialogue blocks into {tgt_lang}. "
            "You will receive a list of dialogue blocks in their correct reading order. "
            "Each block has an index, a speaker name, and an optional context/hint. "
            "Maintain the character relationships, tone, and formatting. "
            "Output the translations in a valid JSON format. "
            "Do NOT output any markdown tags (like ```json), conversational text, or explanations. "
            "Output ONLY the JSON object conforming to this schema:\n"
            "{\n"
            "  \"translations\": [\n"
            "    { \"index\": 1, \"translation\": \"translated text\" },\n"
            "    ...\n"
            "  ]\n"
            "}"
        )
        
        user_prompt = ""
        if global_ctx:
            user_prompt += f"Global Scene Context:\n{global_ctx}\n\n"
        
        user_prompt += "Dialogue blocks to translate:\n"
        for item in dialogues:
            idx = item.get("index")
            text = item.get("text")
            speaker = item.get("speaker")
            ctx = item.get("context")
            
            user_prompt += f"Index: {idx}\n"
            if speaker:
                user_prompt += f"Speaker: {speaker}\n"
            if ctx:
                user_prompt += f"Context: {ctx}\n"
            user_prompt += f"Source Text: {text}\n\n"
        
        yield json.dumps({"type": "progress", "percentage": 0.3, "message": f"Performing translation via {model}..."}) + "\n"
        
        # Execute inference
        if handler_class == "DeepSeekAPI":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")
            
            api_base = os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
            api_base = api_base.rstrip("/")
            url = f"{api_base}/chat/completions"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_cfg.get("model_name", "deepseek-chat"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            if enable_thinking:
                payload["thinking"] = {"type": "enabled"}
                payload["reasoning_effort"] = "high"
            else:
                payload["thinking"] = {"type": "disabled"}
                payload["temperature"] = 0.2
            
            api_start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            res_json = response.json()
            api_elapsed = time.time() - api_start_time
            
            reasoning = res_json["choices"][0]["message"].get("reasoning_content", "")
            if reasoning:
                sys.stderr.write(f"\n[Server Translate Service] DeepSeek Translation Reasoning:\n---\n{reasoning}\n---\n")

            text_final = res_json["choices"][0]["message"]["content"]
            sys.stderr.write(
                f"[Server Translate Service] DeepSeek API translation completed in {api_elapsed:.2f} seconds | "
                f"Response Length: {len(text_final)} characters\n"
            )
        else:
            # Local LLM completion
            llm = get_or_load_model(model)
            llm.reset()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            local_start = time.time()
            response = llm.create_chat_completion(messages=messages)
            local_elapsed = time.time() - local_start
            text_final = response["choices"][0]["message"]["content"]
            sys.stderr.write(
                f"[Server Translate Service] Local LLM ({model}) translation completed in {local_elapsed:.2f} seconds | "
                f"Response Length: {len(text_final)} characters\n"
            )
        
        # Clean up response to get only JSON (some LLMs might wrap in ```json ... ```)
        text_clean = text_final.strip()
        if text_clean.startswith("```"):
            lines = text_clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text_clean = "\n".join(lines).strip()
        
        # Try parsing JSON
        try:
            data = json.loads(text_clean)
            translations_list = data.get("translations", [])
        except Exception as json_err:
            sys.stderr.write(f"[Server Translate Service] Failed to parse translation JSON: {json_err}\nRaw content:\n{text_final}\n")
            # Fallback regex parsing
            translations_list = []
            import re
            matches = re.findall(r'"index"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"([^"]+)"', text_clean)
            for m in matches:
                translations_list.append({"index": int(m[0]), "translation": m[1]})
        
        # Map translations back to the original payload list
        results = []
        trans_dict = {item.get("index"): item.get("translation") for item in translations_list}
        for item in dialogues:
            idx = item.get("index")
            results.append(trans_dict.get(idx, ""))
        
        total_elapsed = time.time() - start_time
        sys.stderr.write(f"[Server Translate Service] Translation processing completed in {total_elapsed:.2f} seconds.\n")
        sys.stderr.write(f"[Server Translate Service] --- Translation Results ---\n")
        for item, res in zip(dialogues, results):
            idx = item.get("index")
            speaker = item.get("speaker") or "Unknown"
            src_text = item.get("text", "").replace("\n", " ")
            res_text = res.replace("\n", " ")
            sys.stderr.write(f"  Block {idx} | Speaker: {speaker} | Source: '{src_text}' -> Translated: '{res_text}'\n")
        sys.stderr.write(f"[Server Translate Service] ---------------------------\n")
        
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": "Done."}) + "\n"
        yield json.dumps({"type": "result", "results": results}) + "\n"
    except Exception as e:
        sys.stderr.write(f"[Server Translate Service] Translation error: {e}\n")
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": f"Error: {e}"}) + "\n"
        yield json.dumps({"type": "result", "results": []}) + "\n"
    finally:
        if handler_class != "DeepSeekAPI":
            unload_model(model)
