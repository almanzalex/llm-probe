### Exploring GPT-2 with English-GUM

This probe explores how GPT-2’s embeddings encode syntactic information, using a sample of the English-GUM corpus. It’s a mix of linguistics and machine learning.

## What it does: 

- GPT-2 Embeddings: We use GPT-2 to generate sentence embeddings from the English-GUM dataset. These embeddings are like fingerprints for each sentence.
- Train the Probe: A probe is a simple model (in this case, linear regression) that helps us see if certain features, like syntax, are encoded in those embeddings.
- Results: By training the probe, we get a sense of which parts of the embeddings are most important and how well they represent syntax.

## embedding_extract.py
- This script extracts embeddings from GPT-2 for the sentences in the English-GUM dataset. It’s like running the dataset through GPT-2 and grabbing the output in a form we can analyze.

## English-gum-reformat.py
- This script prepares the dataset. The English-GUM corpus comes in a specific format, and this script parses it into something cleaner and easy to parse.

## probe.py
- The probe takes the embeddings and checks how well they encode syntactic information by training on them. Visualization of the data is also provided. 

### How to use:
- Start with the English-GUM dataset file. This project includes a script (English-gum-reformat.py) to clean and format it into a JSON file.
- Run the embedding_extract.py script to get embeddings for each sentence in the dataset.
- Use probe.py to train a simple regression model and analyze the embeddings.

### Requirements:

- Python 3
- Libraries: torch, transformers, numpy, scikit-learn, and matplotlib


