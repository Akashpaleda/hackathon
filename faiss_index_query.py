import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load cleaned legal sections
def load_cleaned_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

cleaned_data_file = "cleaned_legal_sections.txt"
data = load_cleaned_data(cleaned_data_file)

# Check if data is valid
if not data:
    print("No valid legal sections found. Check extraction logic.")
    exit()

# Print first 5 extracted legal sections
print("First 5 cleaned legal sections:")
for sec in data[:5]:
    print(f"- {sec}")

# Load pre-trained embedding model
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

# Test Query Retrieval
index = faiss.read_index("legal_faiss.index")
query = "What legal actions can a woman take against domestic violence?"
query_embedding = np.array([embedding_model.encode(query)])
D, I = index.search(query_embedding, k=3)

# Load cleaned legal sections for retrieval
with open(cleaned_data_file, "r", encoding="utf-8") as f:
    legal_sections = f.readlines()

# Display top results
print("Top Relevant Legal Sections (with similarity scores):")
for score, idx in zip(D[0], I[0]):
    if 0 <= idx < len(legal_sections):
        print(f"- [{score:.4f}] {legal_sections[idx].strip()}\n")
