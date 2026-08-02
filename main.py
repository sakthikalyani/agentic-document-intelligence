import os
from utils.pdf_to_image import pdf_to_images
from utils.json_utils import save_json
from agents.planning_agent import PlanningAgent

def run(pdf_path: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("Converting PDF pages to images...")
    image_paths = pdf_to_images(pdf_path, output_dir="data/images")

    print("Loading Planning Agent (Qwen2.5-VL)...")
    agent = PlanningAgent()

    for i, img_path in enumerate(image_paths, start=1):
        print(f"Analyzing page {i}...")
        result = agent.analyze_page(img_path, page_number=i)
        out_path = os.path.join(output_dir, f"page_{i}_plan.json")
        save_json(result, out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    run("data/pdfs/NL1.pdf")