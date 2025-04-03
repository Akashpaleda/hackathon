import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re

# Load and clean legal text
def load_data(file_path):
    """Loads legal sections from a file and ensures proper formatting."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        exit()
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ✅ Normalize "Art. 356" → "Article 356"
    raw_text = re.sub(r'\bArt\.? (\d+)', r'Article \1', raw_text)

    # ✅ Fixed regex: Extracts full section texts without capturing tuples
    data = re.findall(r'\b(?:Article|Section|Schedule) \d+[\.:].*?(?=\b(?:Article|Section|Schedule) \d+[\.:]|\Z)', raw_text.strip(), flags=re.S)

    # ✅ Debug: Print extracted sections
    print(f"\n✅ Extracted {len(data)} legal sections.")

    # 🔍 Show first 5 properly formatted sections
    print("\n🔹 First 5 properly formatted legal sections:")
    for sec in data[:5]:
        print(f"- {sec[:200]}...\n")  # Show first 200 characters

    # 🚨 Error Check: If too few sections found, exit
    if len(data) < 20:
        print("❌ Error: Too few sections extracted! Check the document format.")
        exit()

    return data


# Load full legal document
full_legal_sections_file = "full_legal_sections.txt"
data = load_data(full_legal_sections_file)

# Load pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert legal text into embeddings
embeddings = embedding_model.encode(data, convert_to_numpy=True)

# Validate embeddings
if embeddings.shape[0] < 20:  # Ensure enough embeddings exist
    print("❌ Error: No valid embeddings generated. Exiting.")
    exit()

# Create FAISS index
d = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(d)  # L2 distance metric
index.add(embeddings)

# Save FAISS index and ordered text data
faiss.write_index(index, "legal_faiss.index")
with open("legal_sections_ordered.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(data))

print("\n✅ FAISS index and text data saved successfully!")
print(f"✅ FAISS Index Size: {index.ntotal} entries.")

# Query Function
def query_legal_sections(query, top_k=3):
    """Retrieves the top_k most relevant legal sections for a given query."""
    try:
        # Load FAISS index
        index = faiss.read_index("legal_faiss.index")

        # Convert query to embedding
        query_embedding = embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)

        # Perform search in FAISS index
        D, I = index.search(query_embedding, k=top_k)

        # Load ordered legal sections
        with open("legal_sections_ordered.txt", "r", encoding="utf-8") as f:
            legal_sections = f.readlines()

        print("\n🔍 Top Relevant Legal Sections (with similarity scores):")
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(legal_sections):
                section_text = legal_sections[idx].strip()
                results.append((score, section_text))
                print(f"- [{score:.4f}] {section_text[:200]}...")  # Print first 200 characters
        
        return results
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return []

# Example Query
query = "What legal actions can a woman take against domestic violence?"
query_legal_sections(query)
