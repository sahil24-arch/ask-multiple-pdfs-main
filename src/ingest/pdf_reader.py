from PyPDF2 import PdfReader

def read_pdfs(pdf_files: list) -> str:
    """
    Extract text from a list of uploaded PDF files.
    """
    text = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)
