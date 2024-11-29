from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


app = Flask(__name__)

# Load the model, tokenizer, and FAISS index
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# model.load_state_dict(torch.load("fine_tuned_model.pth"))
# model.load_state_dict(torch.load("fine_tuned_model.pth", map_location=torch.device('cpu')))

model.load_state_dict(torch.load("fine_tuned_model.pth", weights_only=True))

model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# index = faiss.read_index("faiss_index.index")

try:
    index = faiss.read_index("faiss_index.index")
    print("Index is loaded successfully.")
except Exception as e:
    print(f"Failed to load index: {e}")

train_data = pd.read_json('train.jsonl', lines=True)

def clean_code(code):
    code = re.sub(r'\/\*.*?\*\/|\/\/.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()

train_data['clean_code'] = train_data['code'].apply(clean_code)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query_code = request.form['code'].strip()
    print("input is here {}".format(query_code))
    results = retrieve_similar_snippets(query_code)
    print(results)
    return render_template('results.html', results=results)

def retrieve_similar_snippets(query_code, k=5):
    inputs = tokenizer([query_code], truncation=True, padding=True, max_length=512, return_tensors="pt").to(model.device)
    with torch.no_grad():
        query_embedding = model(**inputs).logits.cpu().numpy()
    
    # index.
    print(index)
    print("================================================")
    print(query_embedding)
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "index": idx,
            "code_snippet": train_data.iloc[idx]['clean_code'],
            "distance": distances[0][i]
        })
    return results

if __name__ == '__main__':
    app.run(debug=True)
