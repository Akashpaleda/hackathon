import fitz  # PyMuPDF
import re
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pytesseract
from PIL import Image

def extract_important_sections(pdf_path, output_file):
    doc = fitz.open(pdf_path)
    important_sections = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        # If text is empty, try OCR (for scanned PDFs)
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
        
        # Debugging: Print larger text samples
        if page_num < 3:
            print(f"Page {page_num + 1} Sample Text:\n{text[:1000]}\n---")
        
        # Removing unnecessary line breaks and extra spaces
        text = re.sub(r'\n+', '\n', text).strip()
        
        # More flexible regex pattern
        important_matches = re.findall(r'(Article|Section|IPC)[^\n]*\d+[^\n]*', text)
        
        # Remove empty or amendment-only sections
        important_matches = [sec.strip() for sec in important_matches if len(sec.strip()) > 20 and "amendment" not in sec.lower()]
        
        if important_matches:
            important_sections.extend(important_matches)
    
    # Save cleaned text to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(important_sections))
    
    print(f"Extracted {len(important_sections)} important sections and saved to {output_file}")

# Extract and process the legal text
pdf_path = "inc.pdf"  # Update if needed
output_file = "important_sections.txt"
extract_important_sections(pdf_path, output_file)

# Load extracted text
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

data = load_data(output_file)

# Check if data is empty before proceeding
if not data:
    print("No sections extracted! Check the document format or extraction logic.")
    exit()

# Debug: Print first 5 valid sections
print("First 5 extracted sections:")
for sec in data[:5]:
    print(f"- {sec}")

# Load a pre-trained embedding model
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
faiss.write_index(index, "legal_faiss.index")
print("FAISS index saved successfully!")

# Debug: Print FAISS index size
print(f"FAISS Index Size: {index.ntotal}")

# Test Retrieval
index = faiss.read_index("legal_faiss.index")
query = "What legal actions can a woman take against domestic violence?"
query_embedding = np.array([embedding_model.encode(query)])
D, I = index.search(query_embedding, k=3)

# Load extracted legal sections
with open("important_sections.txt", "r", encoding="utf-8") as f:
    legal_sections = f.readlines()

# Display top results with similarity scores
print("Top Relevant Legal Sections (with similarity scores):")
for score, idx in zip(D[0], I[0]):
    if 0 <= idx < len(legal_sections):
        print(f"- [{score:.4f}] {legal_sections[idx].strip()}\n")
