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

# set padding token to avoid errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# load the parsed GUM dataset from the JSON file
parsed_file_path = "parsed_gum.json"  # adjust if file path differs

with open(parsed_file_path, "r", encoding="utf-8") as file:
    parsed_data = json.load(file)

# extract sentences from parsed data
sentences = [" ".join([token["token"] for token in sentence]) for sentence in parsed_data]

# process sentences in batches to avoid memory issues
batch_size = 2  # adjust batch size as necessary
embeddings = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    
    # tokenize the batch
    inputs = tokenizer(batch_sentences, return_tensors="pt", truncation=True, padding=True)
    
    # forward pass through GPT-2
    with torch.no_grad():
        outputs = model(**inputs)
    
    # extract embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # compute sentence-level embeddings (mean pooling)
    for embedding in last_hidden_state.mean(dim=1):
        embeddings.append(embedding.numpy())
