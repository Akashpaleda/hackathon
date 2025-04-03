import fitz  # PyMuPDF for PDF text extraction
import pytesseract  # OCR for scanned PDFs
from PIL import Image
import re

# Set the path to Tesseract (Modify if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed

def extract_text_from_pdf(pdf_path, output_file):
    """
    Extracts text from a PDF file and saves it to a text file.
    Uses OCR (Tesseract) if the page has no extractable text.
    """
    doc = fitz.open(pdf_path)  # Open the PDF
    extracted_text = []  # Store extracted text

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # Extract text normally
        
        if not text.strip():  # If text extraction fails, try OCR
            print(f"Page {page_num + 1}: No text found, applying OCR...")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)

        # Debugging: Print a sample of extracted text
        print(f"Page {page_num + 1} Extracted Text (First 500 characters):\n{text[:500]}\n---")

        # Store cleaned text
        extracted_text.append(text.strip())

    # Save extracted text to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_text))

    print(f"Extracted text saved to {output_file}")

# Provide the path to your PDF file
pdf_path = "inc.pdf"  # Update if needed
output_file = "extracted_text.txt"

# Run the extraction function
extract_text_from_pdf(pdf_path, output_file)
