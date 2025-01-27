import os
import re
import json

def parse_gum(file_path):
    # parse the English-GUM dataset file and extract sentences, tokens, and annotations.
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            line = line.strip()

            # Skip comments and metadata lines
            if line.startswith("#") or not line:
                if sentence:
                    data.append(sentence)
                    sentence = []
                continue

            # process CoNLL-like lines (token lines)
            parts = line.split('\t')
            if len(parts) >= 10:  # ensure it has at least 10 CoNLL fields
                token = parts[1]  # token text
                label = parts[7]  # dependency label (or customize for other tasks)
                sentence.append({"token": token, "label": label})

    return data

# save parsed data to JSON
def save_to_json(parsed_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(parsed_data, file, indent=4)

if __name__ == "__main__":
    input_file = "english-GUM-sample.txt"  # replace with the actual file path
    output_file = "parsed_gum.json"

    parsed_data = parse_gum(input_file)
    save_to_json(parsed_data, output_file)
    print(f"Parsed data saved to {output_file}")
