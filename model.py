import pandas as pd
import re
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pickle

# ========== 1. Data Preparation ========== #
print("Step 1: Data Preparation")

# Paths to datasets
train_path = 'train.jsonl'
valid_path = 'valid.jsonl'
test_path = 'test.jsonl'

# Load datasets
print("1.1 Loading datasets...")
train_data = pd.read_json(train_path, lines=True)
valid_data = pd.read_json(valid_path, lines=True)
test_data = pd.read_json(test_path, lines=True)
print("Datasets loaded successfully.")

# Function to clean code snippets
def clean_code(code):
    # Remove comments
    code = re.sub(r'\/\*.*?\*\/|\/\/.*?$', '', code, flags=re.MULTILINE)
    # Remove extra blank lines
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()

# Clean datasets
print("1.2 Cleaning datasets...")
train_data['clean_code'] = train_data['code'].apply(clean_code)
valid_data['clean_code'] = valid_data['code'].apply(clean_code)
test_data['clean_code'] = test_data['code'].apply(clean_code)
print("Datasets cleaned successfully.")

# Use only a subset for quick testing
train_data = train_data.head(1000)  # Use the first 1000 rows for training
valid_data = valid_data.head(200)  # Use the first 200 rows for validation
test_data = test_data.head(200)  # Use the first 200 rows for testing

# Split train and validation data
print("1.3 Splitting datasets into training and validation...")
train, validation = train_test_split(train_data, test_size=0.2, random_state=42)
print("Data split successfully.")

# ========== 2. Feature Extraction ========== #
print("Step 2: Feature Extraction")

# Model and tokenizer selection
print("2.1 Loading model and tokenizer...")
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Batch size for processing
BATCH_SIZE = 16

# Generate embeddings in batches
def generate_embeddings(data, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to device
            outputs = model(**inputs)
            batch_embeddings = outputs.logits
            embeddings.append(batch_embeddings.cpu())  # Move to CPU to save memory
    return torch.cat(embeddings, dim=0)

print("2.2 Generating embeddings...")
train_embeddings = generate_embeddings(train['clean_code'].tolist())
valid_embeddings = generate_embeddings(valid_data['clean_code'].tolist())
test_embeddings = generate_embeddings(test_data['clean_code'].tolist())
print("Embeddings generated successfully.")

# Save embeddings
np.save("train_embeddings.npy", train_embeddings.numpy())
np.save("valid_embeddings.npy", valid_embeddings.numpy())
np.save("test_embeddings.npy", test_embeddings.numpy())
print("Embeddings saved successfully.")

# ========== 3. Model Training and Saving ========== #
print("Step 3: Model Training and Saving")

# Prepare data for fine-tuning
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Convert string labels to numeric labels
train['numeric_label'] = pd.factorize(train['label'])[0]  # Creates a numeric label column
validation['numeric_label'] = pd.factorize(validation['label'])[0]  # Ensures consistency for validation

train_encodings = tokenizer(train['clean_code'].tolist(), truncation=True, padding=True, max_length=512)
train_labels = train['numeric_label'].tolist()
train_dataset = CodeDataset(train_encodings, train_labels)

valid_encodings = tokenizer(validation['clean_code'].tolist(), truncation=True, padding=True, max_length=512)
valid_labels = validation['numeric_label'].tolist()
valid_dataset = CodeDataset(valid_encodings, valid_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="no",  # Turn off evaluation
    per_device_train_batch_size=16,  # Increased batch size
    num_train_epochs=0.5,  # Half epoch for quicker validation
    save_strategy="no",  # Turn off checkpoints for quicker runs
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Trainer for fine-tuning
print("3.1 Starting model fine-tuning...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
print("Model fine-tuned successfully.")

# Save the fine-tuned model
print("3.2 Saving the fine-tuned model in multiple formats...")
output_dir = "./fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save in additional formats
torch.save(model.state_dict(), "fine_tuned_model.pth")
pickle.dump(model, open("fine_tuned_model.pkl", "wb"))

print(f"Model saved successfully to {output_dir} in .pth, .pkl, and Hugging Face formats.")

# ========== 4. Building the Retrieval System ========== #
print("Step 4: Building the Retrieval System")

# Convert embeddings to NumPy array
train_embeddings_np = train_embeddings.numpy()

# Create FAISS index
print("4.1 Creating FAISS index...")
index = faiss.IndexFlatL2(train_embeddings_np.shape[1])  # L2 distance
index.add(train_embeddings_np)  # Add embeddings to the index
faiss.write_index(index, "faiss_index.index")
print("FAISS index created and saved.")

# ========== 5. Testing Predictions ========== #
print("Step 5: Testing Predictions")

# Retrieve top K similar snippets
def retrieve_similar_snippets(query_code, k=5):
    inputs = tokenizer([query_code], truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = model(**inputs).logits.cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "index": idx,  # Add index information
            "code_snippet": train.iloc[idx]['clean_code'],
            "distance": distances[0][i]
        })
    return results

# Ask user for input query code
query_code = input("\nEnter your code snippet for similarity search: ").strip()
results = retrieve_similar_snippets(query_code)

# Display results
print(f"\nTop {len(results)} Similar Snippets:")
for result in results:
    print(f"Rank {result['rank']}:")
    print(f"Index: {result['index']}")
    print(f"Code Snippet:\n{result['code_snippet']}")
    print(f"Distance: {result['distance']}\n")

print("Testing completed successfully.")
