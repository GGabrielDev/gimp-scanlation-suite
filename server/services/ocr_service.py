import os
import sys
import json
import base64
import io
import re
import requests
from PIL import Image

from server.core.config import MODELS_CONFIG
from server.services.model_loader import get_or_load_model, unload_model, get_n_gpu_layers_used, set_n_gpu_layers_used
from server.core.prompt_dictionary import PROMPT_DICTIONARY

def preprocess_for_ocr(img_b64: str) -> str:
    """Safely converts the base64 string to an RGB PIL Image and back, without altering the visual data."""
    try:
        header, data = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
        pil_img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        
        # Re-encode to b64 as lossless PNG
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    except Exception as e:
        sys.stderr.write(f"[Server OCR Service] Preprocessing error: {e}\n")
        return img_b64

def parse_arbiter_output(text_final: str) -> tuple[str, str]:
    """
    Robustly parses <thinking> and <transcription> tags from model output.
    Handles common typos (like <transcripition>) and nested tags.
    """
    if not text_final:
        return "", ""

    # 1. Try to find transcription content with regex (handling common typos)
    tx_match = re.search(
        r'<(?:transcription|transcripition|transcribe|output|result)>(.*?)</(?:transcription|transcripition|transcribe|output|result)>',
        text_final,
        re.DOTALL | re.IGNORECASE
    )

    # 2. Try to find thinking content with regex
    think_match = re.search(r'<thinking>(.*?)</thinking>', text_final, re.DOTALL | re.IGNORECASE)

    thinking = ""
    if think_match:
        thinking = think_match.group(1).strip()
    else:
        # Fallback if no closing </thinking> but has opening <thinking>
        think_open = re.search(r'<thinking>(.*)', text_final, re.DOTALL | re.IGNORECASE)
        if think_open:
            content = think_open.group(1)
            # Stop at opening transcription tag if present
            tx_open = re.search(r'<(?:transcription|transcripition|transcribe|output|result)>', content, re.IGNORECASE)
            if tx_open:
                thinking = content[:tx_open.start()].strip()
            else:
                thinking = content.strip()

    transcription = ""
    if tx_match:
        transcription = tx_match.group(1).strip()
    else:
        # Fallback 1: Has opening transcription tag but no closing tag (or cut off)
        tx_open = re.search(r'<(?:transcription|transcripition|transcribe|output|result)>', text_final, re.IGNORECASE)
        if tx_open:
            transcription = text_final[tx_open.end():].strip()
        else:
            # Fallback 2: No transcription tag at all, use text outside thinking
            if think_match:
                remain = text_final.replace(think_match.group(0), "").strip()
                transcription = remain.strip()
            else:
                # Fallback 3: No tags at all, use whole text
                transcription = text_final.strip()

    # Clean up thinking text by removing the transcription content and tags if nested
    if thinking:
        thinking = re.sub(
            r'</?(?:transcription|transcripition|transcribe|output|result)>',
            '',
            thinking,
            flags=re.IGNORECASE
        ).strip()
        if transcription:
            thinking = thinking.replace(transcription, "").strip()

    # Clean up special system/assistant tokens
    if transcription:
        # Remove any <|...|> tokens
        transcription = re.sub(r'<\|.*?\|>', '', transcription).strip()
        # Remove any tags
        transcription = re.sub(r'</?(?:transcription|transcripition|transcribe|output|result|thinking)>', '', transcription, flags=re.IGNORECASE).strip()
        
        # Heuristic for conversational garbage:
        # If there are multiple lines, extract the last non-empty line
        lines = [l.strip() for l in transcription.split('\n') if l.strip()]
        if len(lines) > 1:
            last_line = lines[-1]
            last_line_clean = last_line.strip('"\'「」')
            # If the last line is not conversational instruction and is short, prefer it
            if not any(word in last_line.lower() for word in ["ensure", "transcribe", "output", "thinking", "explanation", "tag", "produce", "data"]):
                transcription = last_line_clean

    return thinking, transcription

def run_single_ocr_generator(model: str, batch_payload: list, options: dict):
    """
    Generator yielding newline-separated JSON progress strings, ending with result payload.
    """
    target_lang = options.get("target_language")
    source_lang = options.get("source_language") or "Japanese"
    material_type = options.get("material_type") or "manga"

    if MODELS_CONFIG.get(model, {}).get("handler_class") == "DeepSeekAPI":
        raw_ocr_model_id = "manga_ocr" if source_lang.lower() == "japanese" else "PaddleOCR_Manga"
        yield json.dumps({
            "type": "progress",
            "percentage": 0.0,
            "message": f"Initializing raw OCR model ({raw_ocr_model_id}) for DeepSeek..."
        }) + "\n"
        try:
            raw_ocr_model = get_or_load_model(raw_ocr_model_id)
        except Exception as e:
            yield json.dumps({"type": "progress", "percentage": 0.0, "message": f"Error: Failed to load raw OCR model: {e}"}) + "\n"
            raise RuntimeError(f"Failed to load raw OCR model: {e}")

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            yield json.dumps({
                "type": "progress",
                "percentage": 0.0,
                "message": "Error: DEEPSEEK_API_KEY environment variable is not set."
            }) + "\n"
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

        llm = None
    else:
        try:
            llm = get_or_load_model(model)
        except Exception as e:
            yield json.dumps({"type": "progress", "percentage": 0.0, "message": f"Error: Failed to load model: {e}"}) + "\n"
            raise RuntimeError(f"Failed to load model: {e}")

    prompt = f"OCR: (Language: {target_lang})" if target_lang else "OCR:"
    sys.stderr.write(f"[Server OCR Service] Processing single-model batch of size {len(batch_payload)} with prompt='{prompt}'...\n")
    results = []
    N = len(batch_payload)

    try:
        for idx, item in enumerate(batch_payload):
            yield json.dumps({
                "type": "progress",
                "percentage": idx / N if N > 0 else 0.0,
                "message": f"Processing crop {idx+1}/{N}..."
            }) + "\n"

            img_str = item.get("image_data", "") if isinstance(item, dict) else str(item)
            if not img_str:
                results.append("")
                continue

            if not img_str.startswith("data:"):
                img_str = f"data:image/png;base64,{img_str}"

            if MODELS_CONFIG.get(model, {}).get("handler_class") == "DeepSeekAPI":
                try:
                    # 1. Run Raw OCR on crop
                    header, data = img_str.split(",", 1)
                    pil_img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
                    
                    if raw_ocr_model_id == "manga_ocr":
                        text_raw = raw_ocr_model(pil_img).strip()
                    else:
                        raw_ocr_model.reset()
                        messages = [
                            {
                                "role": "system",
                                "content": "You are a precise OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not translate, explain, or add conversational filler. If no text is visible, output nothing."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": img_str}},
                                    {"type": "text", "text": "OCR:"}
                                ]
                            }
                        ]
                        response = raw_ocr_model.create_chat_completion(messages=messages)
                        text_raw = response["choices"][0]["message"]["content"].strip()
                    
                    sys.stderr.write(f"[Server OCR Service] DeepSeek Pipeline - Raw OCR text: '{text_raw}'\n")
                    
                    if not text_raw:
                        results.append("")
                        continue

                    # 2. Run DeepSeek API Correction
                    api_base = os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
                    api_base = api_base.rstrip("/")
                    url = f"{api_base}/chat/completions"

                    lang_name = "日本語" if source_lang.lower() == "japanese" else source_lang
                    enable_thinking_global = options.get("enable_thinking", False)
                    enable_thinking_item = item.get("enable_thinking", enable_thinking_global) if isinstance(item, dict) else enable_thinking_global
                    context_hint_item = item.get("context_hint", "") if isinstance(item, dict) else ""

                    if source_lang.lower() == "japanese":
                        system_prompt = (
                            f"あなたは{material_type}の非常に正確なテキスト校正およびOCRポストプロセスのアシスタントです。 "
                            f"あなたの仕事は、提供された原材料から取得した{lang_name}のテキストの誤字脱字やOCR読み取りエラーを校正・修正することです。 "
                            "テキストの翻訳は行わないでください。出力は元の言語の修正後のテキストのみにしてください。 "
                            "説明や対話的な表現、余計なマークダウンは一切含めないでください。"
                        )
                        user_prompt = (
                            f"校正対象の生OCRテキスト:\n"
                            f"\"\"\"\n"
                            f"{text_raw}\n"
                            f"\"\"\"\n\n"
                        )
                        if context_hint_item:
                            user_prompt += f"このテキストの追加の文脈情報/ヒント: {context_hint_item}\n\n"
                        user_prompt += (
                            f"元の意味とレイアウトを維持したまま、修正された{lang_name}のテキストのみを出力してください。 "
                            "テキストがすでに正確であるか、修正が必要ない場合は、生OCRテキストをそのまま出力してください。"
                        )
                    else:
                        system_prompt = (
                            f"You are a precise text editor and OCR post-processing assistant for {material_type}. "
                            f"Your job is to correct and refine raw OCR text transcribed from the source language {lang_name}. "
                            "Do NOT translate the text. Output ONLY the corrected text in the original language. "
                            "Do not add any explanations, markdown formatting, or conversational filler."
                        )
                        user_prompt = (
                            f"Raw OCR text to correct:\n"
                            f"\"\"\"\n"
                            f"{text_raw}\n"
                            f"\"\"\"\n\n"
                        )
                        if context_hint_item:
                            user_prompt += f"Additional Context Hint for this text: {context_hint_item}\n\n"
                        user_prompt += (
                            f"Please output ONLY the corrected {lang_name} text, maintaining the original meaning and layout. "
                            f"If the text is already correct or there is nothing to correct, output the raw text as-is."
                        )

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": MODELS_CONFIG.get(model, {}).get("model_name", "deepseek-chat"),
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    }
                    if enable_thinking_item:
                        payload["thinking"] = {"type": "enabled"}
                        payload["reasoning_effort"] = "high"
                    else:
                        payload["thinking"] = {"type": "disabled"}
                        payload["temperature"] = 0.2
                    
                    response = requests.post(url, json=payload, headers=headers, timeout=120.0)
                    response.raise_for_status()
                    res_json = response.json()
                    
                    reasoning = res_json["choices"][0]["message"].get("reasoning_content", "")
                    if reasoning:
                        sys.stderr.write(f"\n[Server OCR Service] DeepSeek Reasoning:\n---\n{reasoning}\n---\n")

                    corrected_text = res_json["choices"][0]["message"]["content"].strip()
                    corrected_text = corrected_text.strip().strip('"\'')
                    sys.stderr.write(f"[Server OCR Service] DeepSeek Pipeline - Corrected text: '{corrected_text}'\n")
                    results.append(corrected_text)
                    
                except Exception as ocr_err:
                    sys.stderr.write(f"[Server OCR Service] Error during DeepSeek OCR pipeline on item {idx}: {ocr_err}\n")
                    results.append(text_raw if 'text_raw' in locals() else "")
            else:
                try:
                    llm.reset()
                except Exception as r_err:
                    sys.stderr.write(f"[Server OCR Service] LLM reset error: {r_err}\n")

                if MODELS_CONFIG[model].get("handler_class") == "Llava15ChatHandler":
                    user_text = f"You are a precise OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not translate, explain, or add conversational filler. If no text is visible, output nothing.\n\nPrompt: {prompt}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_str}},
                                {"type": "text", "text": user_text}
                            ]
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a precise OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not translate, explain, or add conversational filler. If no text is visible, output nothing."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_str}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]

                try:
                    try:
                        response = llm.create_chat_completion(messages=messages)
                    except Exception as inf_err:
                        if get_n_gpu_layers_used() != 0:
                            sys.stderr.write(f"[Server OCR Service] GPU execution failed: {inf_err}. Re-routing to CPU...\n")
                            llm = get_or_load_model(model, force_cpu=True)
                            llm.reset()
                            response = llm.create_chat_completion(messages=messages)
                        else:
                            raise inf_err

                    text = response["choices"][0]["message"]["content"]
                    results.append(text.strip())
                except Exception as e:
                    sys.stderr.write(f"[Server OCR Service] Error during OCR inference on item {idx}: {e}\n")
                    results.append("")

        yield json.dumps({
            "type": "progress",
            "percentage": 1.0,
            "message": "Completed OCR batch."
        }) + "\n"
        yield json.dumps({"type": "result", "results": results}) + "\n"
    finally:
        if MODELS_CONFIG.get(model, {}).get("handler_class") == "DeepSeekAPI":
            unload_model(raw_ocr_model_id)
        else:
            unload_model(model)

def run_ensemble_ocr_generator(model: str, batch_payload: list, options: dict):
    """
    Generator yielding consensus (ensemble) OCR operations.
    """
    source_lang = options.get("source_language") or "Japanese"
    if source_lang not in ["Japanese", "English"]:
        source_lang = "Japanese"
    material_type = options.get("material_type") or "manga"

    system_instruction = PROMPT_DICTIONARY.get(material_type, PROMPT_DICTIONARY["manga"])

    arbiter_model_id = options.get("consensus_arbiter") or model
    if arbiter_model_id == "Ensemble":
        arbiter_model_id = "DeepSeek"

    if arbiter_model_id not in MODELS_CONFIG:
        raise ValueError(f"Arbiter VLM Model ID '{arbiter_model_id}' is not registered.")

    expert_b_model_id = options.get("consensus_expert_b") or "PaddleOCR_Manga"

    crops_base64 = []
    for item in batch_payload:
        img_str = item.get("image_data", "") if isinstance(item, dict) else str(item)
        if not img_str:
            crops_base64.append("")
            continue
        if not img_str.startswith("data:"):
            img_str = f"data:image/png;base64,{img_str}"
        crops_base64.append(img_str)

    crops_base64 = [preprocess_for_ocr(crop) if crop else "" for crop in crops_base64]
    sys.stderr.write(f"[Server OCR Service] Starting Ensemble OCR Consensus on {len(crops_base64)} crops...\n")

    enable_thinking = options.get("enable_thinking", False)
    N = len(crops_base64)
    if N == 0:
        yield json.dumps({"type": "progress", "percentage": 1.0, "message": "No crops to process."}) + "\n"
        yield json.dumps({"type": "result", "results": []}) + "\n"
        return

    # --- PASS 1: manga-ocr (PyTorch) ---
    results_a = []
    try:
        yield json.dumps({
            "type": "progress",
            "percentage": 0.0,
            "message": "Initializing PyTorch manga-ocr..."
        }) + "\n"
        mocr = get_or_load_model("manga_ocr")
        for idx, img_b64 in enumerate(crops_base64):
            pct = idx / (3 * N)
            yield json.dumps({
                "type": "progress",
                "percentage": pct,
                "message": f"Pass 1/3 (manga-ocr): Crop {idx+1}/{N}..."
            }) + "\n"
            
            if not img_b64:
                results_a.append("")
                continue
            try:
                header, data = img_b64.split(",", 1)
                pil_img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
                text_a = mocr(pil_img)
                results_a.append(text_a.strip() if text_a else "")
            except Exception as ex_a:
                sys.stderr.write(f"[Server OCR Service] Expert A (manga-ocr) error on crop {idx}: {ex_a}\n")
                results_a.append("")
    finally:
        unload_model("manga_ocr")

    # --- PASS 2: Expert B ---
    results_b = []
    is_vision_model = MODELS_CONFIG.get(expert_b_model_id, {}).get("handler_class") not in ["TextOnly", "DeepSeekAPI"]
    if not is_vision_model:
        expert_b_model_id = "PaddleOCR_Manga"
    try:
        yield json.dumps({
            "type": "progress",
            "percentage": N / (3 * N),
            "message": f"Initializing Expert B ({expert_b_model_id})..."
        }) + "\n"
        expert_b = get_or_load_model(expert_b_model_id)
        for idx, img_b64 in enumerate(crops_base64):
            pct = (N + idx) / (3 * N)
            yield json.dumps({
                "type": "progress",
                "percentage": pct,
                "message": f"Pass 2/3 (Expert B VLM): Crop {idx+1}/{N}..."
            }) + "\n"
            
            if not img_b64:
                results_b.append("")
                continue
            try:
                expert_b.reset()
                messages = [
                    {
                        "role": "system",
                        "content": "You are a precise Japanese OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not translate, explain, or add conversational filler. If no text is visible, output nothing."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_b64}},
                            {"type": "text", "text": "OCR:"}
                        ]
                    }
                ]
                try:
                    response = expert_b.create_chat_completion(messages=messages)
                except Exception as inf_err:
                    if get_n_gpu_layers_used() != 0:
                        sys.stderr.write(f"[Server OCR Service] Expert B GPU failed. Re-routing to CPU...\n")
                        expert_b = get_or_load_model(expert_b_model_id, force_cpu=True)
                        expert_b.reset()
                        response = expert_b.create_chat_completion(messages=messages)
                    else:
                        raise inf_err

                text_b = response["choices"][0]["message"]["content"]
                results_b.append(text_b.strip() if text_b else "")
            except Exception as ex_b:
                sys.stderr.write(f"[Server OCR Service] Expert B ({expert_b_model_id}) error on crop {idx}: {ex_b}\n")
                results_b.append("")
    finally:
        if expert_b_model_id != arbiter_model_id:
            unload_model(expert_b_model_id)

    # --- PASS 3: Arbiter VLM Consensus ---
    final_results = []
    try:
        yield json.dumps({
            "type": "progress",
            "percentage": (2 * N) / (3 * N),
            "message": f"Initializing Arbiter ({arbiter_model_id})..."
        }) + "\n"
        arbiter = get_or_load_model(arbiter_model_id)
        for idx, img_b64 in enumerate(crops_base64):
            pct = (2 * N + idx) / (3 * N)
            yield json.dumps({
                "type": "progress",
                "percentage": pct,
                "message": f"Pass 3/3 (Arbiter VLM): Crop {idx+1}/{N}..."
            }) + "\n"
            
            if not img_b64:
                final_results.append("")
                continue
            
            result_a = results_a[idx]
            result_b = results_b[idx]

            try:
                reasoning = ""
                item = batch_payload[idx] if isinstance(batch_payload[idx], dict) else {}
                enable_thinking_item = item.get("enable_thinking", enable_thinking)
                context_hint_item = item.get("context_hint", "")

                if source_lang == "Japanese":
                    user_prompt = (
                        f"【専門OCRモデルによる読み取りデータ】\n"
                        f"- データ A: {result_a}\n"
                        f"- データ B: {result_b}\n\n"
                        f"素材タイプ (Material Type): {material_type}\n\n"
                    )
                    if context_hint_item:
                        user_prompt += f"このテキストの追加の文脈情報/ヒント: {context_hint_item}\n\n"
                    user_prompt += (
                        f"あなたはOCRエラーを修正する専門家です。提供されたデータを単に比較するのではなく、これらをベースとして使用し、指定された素材タイプの文脈、文法、および一般的なOCRの弱点（文字の欠落や誤読）を考慮して、最も正確なテキストを推論してください。\n\n"
                        f"以下のステップで推論を行ってください：\n"
                        f"1. データの統合: 両方のデータを分析し、素材の文脈（例：スラング、擬音語、特殊なフォーマット）に最も適した文字や単語を抽出します。\n"
                        f"2. エラー修正: 視覚モデルがよく間違う文字（例：「ン」の欠落、濁点・半濁点の誤り、小さな仮名）を論理的に修正します。\n\n"
                        f"ステップバイステップの推論を <thinking>...</thinking> タグ内に記述し、最終的な修正済み日本語テキストを <transcription>...</transcription> タグ内に記述してください。\n\n"
                        f"<thinking>\n[あなたの推論]</thinking>\n<transcription>\n[最終的なテキストのみ]</transcription>"
                    )
                else:
                    user_prompt = (
                        f"[Raw OCR Data]\n"
                        f"- Data A: {result_a}\n"
                        f"- Data B: {result_b}\n\n"
                        f"Material Type: {material_type}\n\n"
                    )
                    if context_hint_item:
                        user_prompt += f"Additional Context Hint for this text: {context_hint_item}\n\n"
                    user_prompt += (
                        f"You are an expert OCR correction engine. Do not just pick between Data A and Data B. Use them as a baseline to infer the perfectly accurate transcription based on the specific context of the material type.\n\n"
                        f"Protocol:\n"
                        f"1. Synthesis: Analyze both readings to extract the most logical words based on the material's tone and formatting (e.g., comic book block lettering, sound effects).\n"
                        f"2. Correction: Fix common OCR artifacts, hallucinated characters, and punctuation errors to form a coherent string.\n\n"
                        f"Write your step-by-step reasoning inside <thinking>...</thinking> tags, and the final corrected transcription inside <transcription>...</transcription> tags.\n\n"
                        f"<thinking>\n[Your reasoning]</thinking>\n<transcription>\n[Final text only]</transcription>"
                    )

                if MODELS_CONFIG.get(arbiter_model_id, {}).get("handler_class") == "DeepSeekAPI":
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
                        "model": MODELS_CONFIG.get(arbiter_model_id, {}).get("model_name", "deepseek-chat"),
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ]
                    }
                    if enable_thinking:
                        payload["thinking"] = {"type": "enabled"}
                        payload["reasoning_effort"] = "high"
                    else:
                        payload["thinking"] = {"type": "disabled"}
                        payload["temperature"] = 0.2
                    
                    response = requests.post(url, json=payload, headers=headers, timeout=120.0)
                    response.raise_for_status()
                    res_json = response.json()
                    
                    reasoning = res_json["choices"][0]["message"].get("reasoning_content", "")
                    if reasoning:
                        sys.stderr.write(f"\n[Server OCR Service] DeepSeek Arbiter Reasoning:\n---\n{reasoning}\n---\n")

                    text_final = res_json["choices"][0]["message"]["content"]
                else:
                    arbiter.reset()
                    
                    if MODELS_CONFIG[arbiter_model_id].get("handler_class") == "TextOnly":
                        messages = [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ]
                    elif MODELS_CONFIG[arbiter_model_id].get("handler_class") == "Llava15ChatHandler":
                        user_text = f"{system_instruction}\n\n{user_prompt}"
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_text}
                                ]
                            }
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": system_instruction},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt}
                                ]
                            }
                        ]

                    try:
                        response = arbiter.create_chat_completion(messages=messages, temperature=0.2, max_tokens=2048)
                    except Exception as inf_err:
                        if get_n_gpu_layers_used() != 0:
                            sys.stderr.write(f"[Server OCR Service] Arbiter VLM GPU execution failed. Re-routing to CPU...\n")
                            arbiter = get_or_load_model(arbiter_model_id, force_cpu=True)
                            arbiter.reset()
                            response = arbiter.create_chat_completion(messages=messages, temperature=0.2, max_tokens=2048)
                        else:
                            raise inf_err

                    text_final = response["choices"][0]["message"]["content"]
                
                thinking, transcription = parse_arbiter_output(text_final)
                
                if MODELS_CONFIG.get(arbiter_model_id, {}).get("handler_class") == "DeepSeekAPI" and enable_thinking_item:
                    if reasoning:
                        thinking = reasoning

                if thinking:
                    sys.stderr.write(f"\n[Server OCR Service] Crop {idx} Thinking:\n---\n{thinking}\n---\n")
                    
                final_results.append(transcription)
                sys.stderr.write(f"[Server OCR Service] Ensemble crop {idx} final: '{result_a}' / '{result_b}' -> '{transcription}'\n")
            except Exception as ex_c:
                sys.stderr.write(f"[Server OCR Service] Arbiter consensus error on crop {idx}: {ex_c}\n")
                final_results.append(result_a or result_b or "")
            
    finally:
        if arbiter_model_id != "DeepSeek":
            unload_model(arbiter_model_id)

    yield json.dumps({
        "type": "progress",
        "percentage": 1.0,
        "message": "Completed Ensemble OCR Consensus."
    }) + "\n"
    yield json.dumps({"type": "result", "results": final_results}) + "\n"
