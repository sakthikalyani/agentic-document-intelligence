# import torch
# from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# MODEL_NAME = "google/gemma-3-4b-it"

# _model = None
# _processor = None

# def get_gemma():
#     global _model, _processor
#     if _model is None:
#         print("Loading Gemma 3 4B (first time only)...")
#         _model = Gemma3ForConditionalGeneration.from_pretrained(
#             MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
#         )
#         _processor = AutoProcessor.from_pretrained(MODEL_NAME)
#         print("Gemma 3 loaded.")
#     else:
#         print("Reusing already-loaded Gemma model.")
#     return _model, _processor

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

_model = None
_processor = None

def get_qwen():
    global _model, _processor
    if _model is None:
        print("Loading Qwen2.5-VL-7B (first time only)...")
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        print("Qwen loaded.")
    else:
        print("Reusing already-loaded Qwen model.")
    return _model, _processor