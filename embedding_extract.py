import json
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Set padding token to avoid errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Load the parsed English-GUM dataset from the JSON file
parsed_file_path = "parsed_gum.json"  # Replace with your English-GUM JSON file path

with open(parsed_file_path, "r", encoding="utf-8") as file:
    parsed_data = json.load(file)

# Extract sentences and their labels from parsed data
sentences = [" ".join([token["token"] for token in sentence]) for sentence in parsed_data]
labels = [sentence[0]["label"] for sentence in parsed_data]  # Assuming each sentence has a label

# Process sentences in batches to avoid memory issues
batch_size = 2  # Adjust batch size as necessary
embeddings = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    
    # Tokenize the batch
    inputs = tokenizer(batch_sentences, return_tensors="pt", truncation=True, padding=True)
    
    # Forward pass through GPT-2
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # Compute sentence-level embeddings (mean pooling)
    for embedding in last_hidden_state.mean(dim=1):
        embeddings.append(embedding.numpy())

# Save embeddings and labels to a JSON file
output_data = [
    {
        "embedding": embeddings[i].tolist(),
        "label": labels[i],
    }
    for i in range(len(labels))
]

with open("gum_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)
