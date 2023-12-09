import os
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from pathlib import Path

models_directory = "./"

def load_model(model_name):
    model_path = os.path.join(models_directory, model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    return model, tokenizer

def classify_code(file_path, model_name):
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)

    # Read the code from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()

    # Tokenize the code
    inputs = tokenizer(code, return_tensors="pt", truncation=True)

    # Make prediction
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).item()

    return predictions

def main(file_path):
    # Get the file extension
    file_extension = Path(file_path).suffix[1:]  # Exclude the dot from the extension

    # Construct the model name based on the file extension
    model_name = f"{file_extension.lower()}_model"
    
    # Try to load and classify using the corresponding model
    try:
        prediction = classify_code(file_path, model_name)
        print(f"File '{file_path}' is {'vulnerable' if prediction == 1 else 'secure'}")
    except FileNotFoundError:
        print(f"Model for {file_extension} not found. {file_extension} not supported.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python classify.py <file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)
