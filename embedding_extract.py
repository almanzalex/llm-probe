import json
import torch
from transformers import GPT2Tokenizer, GPT2Model

# load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# set padding token to avoid errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# path to parsed English-GUM dataset
parsed_file_path = "parsed_gum.json"  # Replace with your English-GUM JSON file path

# load parsed data
with open(parsed_file_path, "r", encoding="utf-8") as file:
    parsed_data = json.load(file)

# define a label mapping (update as needed for your dataset)
label_mapping = {
    "root": 0,
    "dep": 1,
    "advmod": 2,
    # add more label mappings as needed
}

# extract sentences and their labels
sentences = [" ".join([token["token"] for token in sentence]) for sentence in parsed_data]
labels = [label_mapping.get(sentence[0]["label"], -1) for sentence in parsed_data]  # Map labels to integers

# filter out sentences with unknown labels
filtered_data = [
    (sent, label)
    for sent, label in zip(sentences, labels)
    if label != -1  # exclude sentences with unmapped labels
]

# split filtered data into sentences and labels
filtered_sentences, filtered_labels = zip(*filtered_data)

# process sentences in batches to avoid memory issues
batch_size = 2  # adjust batch size as necessary
embeddings = []

for i in range(0, len(filtered_sentences), batch_size):
    batch_sentences = filtered_sentences[i:i + batch_size]
    
    # tokenize the batch
    inputs = tokenizer(batch_sentences, return_tensors="pt", truncation=True, padding=True)
    
    # forward pass through GPT-2
    with torch.no_grad():
        outputs = model(**inputs)
    
    # extract embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # compute sentence-level embeddings (mean)
    for embedding in last_hidden_state.mean(dim=1):
        embeddings.append(embedding.numpy())

# save embeddings and labels to a JSON file
output_data = [
    {
        "embedding": embeddings[i].tolist(),
        "label": filtered_labels[i],
    }
    for i in range(len(filtered_labels))
]

output_file_path = "gum_embeddings.json"  # output path for embeddings
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)

print(f"Embeddings saved to {output_file_path}")
