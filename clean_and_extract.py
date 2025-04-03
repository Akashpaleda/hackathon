import re

# Load extracted text
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

# Extract full legal sections
def extract_legal_sections(data):
    cleaned_sections = []
    
    for line in data:
        line = line.strip()

        # Improved regex to capture complete legal references
        match = re.search(r'(Article|Section|Act|IPC) \d+(\(\d+\))?( of [\w\s]+)?', line, re.IGNORECASE)
        
        if match:
            cleaned_text = match.group(0).strip()

            # Ensure full references (skip short or incomplete ones)
            if len(cleaned_text.split()) < 4:  # Less than 4 words means incomplete reference
                continue  

            # Exclude unwanted words
            if any(keyword in cleaned_text.lower() for keyword in ["omitted", "footnote", "repealed", "act as"]):
                continue  
            
            cleaned_sections.append(cleaned_text)

    return cleaned_sections

# File names
extracted_text_file = "extracted_text.txt"
cleaned_text_file = "cleaned_legal_sections.txt"

# Load and clean the data
raw_data = load_data(extracted_text_file)
legal_sections = extract_legal_sections(raw_data)

# Save cleaned legal sections
with open(cleaned_text_file, "w", encoding="utf-8") as f:
    f.write("\n".join(legal_sections))

# Debug: Print first 5 cleaned sections
print("First 5 cleaned legal sections:")
for sec in legal_sections[:5]:
    print(f"- {sec}")

print(f"\nTotal cleaned sections extracted: {len(legal_sections)}")
print(f"Cleaned legal sections saved to {cleaned_text_file}")
