# import fitz
# from pathlib import Path


# class PDFToImage:

#     @staticmethod
#     def convert(pdf_path: str, output_dir: str):

#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         pdf = fitz.open(pdf_path)

#         pages = []

#         for page_number, page in enumerate(pdf):

#             pix = page.get_pixmap(dpi=200)

#             image_path = output_dir / f"page_{page_number + 1}.png"

#             pix.save(str(image_path))

#             pages.append(str(image_path))

#         pdf.close()

#         return pages
import fitz  # PyMuPDF
from PIL import Image
import io
import os

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    """Convert each page of a PDF into a PNG image. Returns list of image paths."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        out_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        img.save(out_path)
        image_paths.append(out_path)

    doc.close()
    return image_paths