import fitz  # PyMuPDF
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image

def extract_full_legal_text(pdf_path, output_file):
    doc = fitz.open(pdf_path)
    legal_sections = []
    section_text = ""
    section_title = ""
    
    for page in doc:
        text = page.get_text("text")
        
        # OCR fallback for scanned PDFs
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
        
        text = re.sub(r'\n+', '\n', text).strip()
        
        lines = text.split("\n")
        for line in lines:
            match = re.match(r'(Article|Section|IPC)\s*(\d+).*', line, re.IGNORECASE)
            if match:
                if section_title and section_text:
                    legal_sections.append((section_title, section_text.strip()))
                section_title = line.strip()
                section_text = ""
            else:
                section_text += " " + line.strip()
        
    if section_title and section_text:
        legal_sections.append((section_title, section_text.strip()))
    
    with open(output_file, "w", encoding="utf-8") as f:
        for title, content in legal_sections:
            f.write(f"{title}: {content}\n\n")
    
    print(f"Extracted {len(legal_sections)} full legal sections and saved to {output_file}")

# File paths
pdf_path = "inc.pdf"  # Update this to your actual PDF file
output_file = "full_legal_sections.txt"
extract_full_legal_text(pdf_path, output_file)

# Load extracted full sections
def load_full_legal_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

data = load_full_legal_data(output_file)

# Check if data is valid
if not data:
    print("No full legal sections extracted! Check the extraction logic.")
    exit()

print("First 5 extracted full legal sections:")
for sec in data[:5]:
    print(f"- {sec}")

# Load sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array(embedding_model.encode(data))

if embeddings.size == 0:
    print("Error: No valid embeddings generated. Exiting.")
    exit()

# Create FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "legal_faiss_full.index")
print("FAISS index with full legal text saved successfully!")