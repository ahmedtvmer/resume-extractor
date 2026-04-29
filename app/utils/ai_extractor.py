"""
AI-powered resume field extractor.

Loads a fine-tuned TinyLlama + LoRA model and uses it to extract
Name, Email Address, Skills, and Education from raw resume text.
"""

import json
import logging
import os
import re
import asyncio
from functools import lru_cache
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = os.getenv("RESUME_MODEL_PATH", "./final-resume-model")
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 256

# Fields the model was trained to extract
FIELDS = ["Name", "Email Address", "Skills", "Education"]


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
class _ModelHolder:
    """Holds the model & tokenizer as a process-wide singleton."""

    model: Optional[PeftModel] = None
    tokenizer: Optional[AutoTokenizer] = None
    device: Optional[str] = None


_holder = _ModelHolder()


def load_model() -> None:
    """
    Load the base TinyLlama model + LoRA adapters into memory.
    Called once at application startup via the FastAPI lifespan hook.
    """
    if _holder.model is not None:
        logger.info("Model already loaded – skipping.")
        return

    logger.info("Loading base model: %s", BASE_MODEL_ID)

    # Determine compute device
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
        _holder.device = "cuda"
        logger.info("CUDA available – using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device_map = "cpu"
        dtype = torch.float32
        _holder.device = "cpu"
        logger.info("No GPU found – running on CPU (inference will be slower).")

    # Load tokenizer
    _holder.tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    _holder.tokenizer.pad_token = _holder.tokenizer.eos_token
    _holder.tokenizer.padding_side = "right"

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Merge LoRA adapters
    adapter_path = os.path.abspath(ADAPTER_PATH)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"LoRA adapter directory not found at '{adapter_path}'. "
            "Download the fine-tuned model or set RESUME_MODEL_PATH env var."
        )

    logger.info("Loading LoRA adapters from: %s", adapter_path)
    _holder.model = PeftModel.from_pretrained(base_model, adapter_path)
    _holder.model.eval()

    logger.info("✅ Resume extraction model loaded successfully.")


def unload_model() -> None:
    """Release model from memory (called on shutdown)."""
    _holder.model = None
    _holder.tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _build_prompt(resume_text: str) -> str:
    """Build the prompt in the same format used during training."""
    content = resume_text.replace("\n", " ").strip()
    return (
        "### Instruction: Extract Name, Email Address, Skills, "
        "and Education from the resume.\n"
        f"### Input: {content}\n"
        "### Response:"
    )


def _parse_model_output(raw_output: str) -> Dict[str, str]:
    """
    Parse the raw model output into a dict of extracted fields.
    The model was trained to emit JSON after '### Response:'.
    Falls back to regex extraction if JSON parsing fails.
    """
    # Isolate the response portion
    if "### Response:" in raw_output:
        response_text = raw_output.split("### Response:")[-1].strip()
    else:
        response_text = raw_output.strip()

    # Strip markdown code fences if present (e.g., ```json ... ```)
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()

    # Attempt 1: Direct JSON parse
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return {field: parsed.get(field, "") or "" for field in FIELDS}
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: Find JSON-like substring with regex
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return {field: parsed.get(field, "") or "" for field in FIELDS}
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 3: Fallback – return the raw text in a best-effort dict
    logger.warning("Could not parse model output as JSON. Raw: %s", response_text[:300])
    return {field: "" for field in FIELDS}


async def extract_fields(resume_text: str) -> Dict[str, str]:
    """
    Run inference on the loaded model and return extracted fields.

    Returns a dict like:
        {
            "Name": "...",
            "Email Address": "...",
            "Skills": "...",
            "Education": "..."
        }
    """
    if _holder.model is None or _holder.tokenizer is None:
        raise RuntimeError(
            "Model is not loaded. Ensure load_model() was called at startup."
        )

    def _run_inference():
        prompt = _build_prompt(resume_text)

        inputs = _holder.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(_holder.model.device)

        # Log warning if prompt was truncated
        if inputs["input_ids"].shape[1] == MAX_INPUT_TOKENS:
            logger.warning(
                "Resume text was truncated to %d tokens during inference. "
                "Some content may have been lost.",
                MAX_INPUT_TOKENS
            )

        with torch.no_grad():
            outputs = _holder.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.1,
            )

        full_output = _holder.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _parse_model_output(full_output)

    return await asyncio.to_thread(_run_inference)


if __name__ == "__main__":
    """
    Run unit tests for _parse_model_output().
    Usage: python -m app.utils.ai_extractor --test
    """
    import sys
    
    if "--test" in sys.argv:
        print("Running unit tests for _parse_model_output()...\n")
        
        # Test 1: Valid JSON
        print("Test 1: Valid JSON parsing")
        output = '{"Name": "John Doe", "Email Address": "john@example.com", "Skills": "Python, SQL", "Education": "BS CS"}'
        result = _parse_model_output(output)
        assert result["Name"] == "John Doe"
        assert result["Email Address"] == "john@example.com"
        assert result["Skills"] == "Python, SQL"
        assert result["Education"] == "BS CS"
        print("PASSED\n")
        
        # Test 2: JSON with null values
        print("Test 2: JSON with null values")
        output = '{"Name": null, "Email Address": "test@test.com", "Skills": null, "Education": ""}'
        result = _parse_model_output(output)
        assert result["Name"] == ""
        assert result["Email Address"] == "test@test.com"
        assert result["Skills"] == ""
        assert result["Education"] == ""
        print("PASSED\n")
        
        # Test 3: Code-fenced JSON
        print("Test 3: Code-fenced JSON (```json ... ```)")
        output = '```json\n{"Name": "Jane", "Email Address": "jane@test.com", "Skills": "Java", "Education": "MS"}\n```'
        result = _parse_model_output(output)
        assert result["Name"] == "Jane"
        assert result["Email Address"] == "jane@test.com"
        print("PASSED\n")
        
        # Test 4: JSON embedded in text (regex fallback)
        print("Test 4: JSON embedded in text (regex extraction)")
        output = 'Here is the result: {"Name": "Bob", "Email Address": "bob@test.com", "Skills": "C++", "Education": "PhD"} extracted.'
        result = _parse_model_output(output)
        assert result["Name"] == "Bob"
        assert result["Email Address"] == "bob@test.com"
        print("PASSED\n")
        
        # Test 5: Malformed output (fallback to empty)
        print("Test 5: Malformed output (fallback to empty dict)")
        output = 'This is just random text with no JSON at all'
        result = _parse_model_output(output)
        assert result == {"Name": "", "Email Address": "", "Skills": "", "Education": ""}
        print("PASSED\n")
        
        # Test 6: With ### Response: prefix
        print("Test 6: Output with ### Response: prefix")
        output = '### Instruction: ...\n### Input: ...\n### Response: {"Name": "Alice", "Email Address": "alice@test.com", "Skills": "Rust", "Education": "BS"}'
        result = _parse_model_output(output)
        assert result["Name"] == "Alice"
        assert result["Email Address"] == "alice@test.com"
        print("PASSED\n")
        
        print("All unit tests passed!")
        sys.exit(0)
