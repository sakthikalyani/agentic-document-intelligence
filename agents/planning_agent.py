import os
from utils.json_utils import extract_json
from agents.model_loader import get_qwen, MIN_PIXELS, MAX_PIXELS

class PlanningAgent:
    def __init__(self):
        self.model, self.processor = get_qwen()

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "planning_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def analyze_page(self, image_path: str, page_number: int) -> dict:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": self.prompt_template},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=1536, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        result = extract_json(output_text)
        result["page"] = page_number
        return result
# import os
# import torch
# from utils.json_utils import extract_json
# from agents.model_loader import get_gemma

# class PlanningAgent:
#     def __init__(self):
#         self.model, self.processor = get_gemma()

#         prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "planning_prompt.txt")
#         with open(prompt_path, "r", encoding="utf-8") as f:
#             self.prompt_template = f.read()

#     def analyze_page(self, image_path: str, page_number: int) -> dict:
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image_path},
#                     {"type": "text", "text": self.prompt_template},
#                 ],
#             }
#         ]

#         inputs = self.processor.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt"
#         ).to(self.model.device, dtype=torch.bfloat16)

#         input_len = inputs["input_ids"].shape[-1]

#         with torch.inference_mode():
#             output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
#             output_ids = output_ids[0][input_len:]

#         output_text = self.processor.decode(output_ids, skip_special_tokens=True)

#         result = extract_json(output_text)
#         result["page"] = page_number
#         return result