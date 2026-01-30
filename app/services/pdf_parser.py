from pypdf import PdfReader
from io import BytesIO


def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_stream = BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text
