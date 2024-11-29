Here's a corrected version of the `README.md` file with better alignment and formatting to ensure consistency across sections and improve readability:

```markdown
# Code Snippet Similarity Search

This project implements a Flask-based web application for retrieving semantically similar code snippets. It uses a machine learning model to generate embeddings for code snippets and employs a FAISS index for efficient similarity searches.

## Project Structure

/flask_app
│
├── /fine_tuned_model
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── /static
│   ├── /css
│   │   └── style.css
│   ├── /js
│   └── /images
│
├── /templates
│   ├── index.html
│   └── results.html
│
├── app.py
├── model.py
├── faiss_index.index
├── fine_tuned_model.pth
├── fine_tuned_model.pkl
├── test_embeddings.npy
├── train_embeddings.npy
├── valid_embeddings.npy
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip3

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yuvi-crypto/Semantically-similar-code-snippet-Clones-retrieval.git
   cd Semantically-similar-code-snippet-Clones-retrieval
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

This will start the Flask server on `http://localhost:5000`, and you can navigate to this address in your web browser to use the application.

## Usage

1. **Home Page:**
   - Access the home page at `http://localhost:5000`.
   - Enter a code snippet in the provided text area.

2. **Search Similar Code Snippets:**
   - Click the 'Find Similar Code' button after entering your snippet.
   - The system will display the top similar code snippets based on semantic similarity.

## Features

- **Code Snippet Input**: Users can input any code snippet to find similar code snippets.
- **Semantic Similarity Search**: Uses advanced NLP models to understand the semantics of the code.
- **FAISS Indexing**: Leverages FAISS for fast and efficient similarity search.

## Contributing

Contributions to the project are welcome! Please refer to the following steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a new Pull Request.


For any queries, you can reach out to [yuvaraj.gajalajamgam@students.iiit.ac.in](mailto:yuvaraj.gajalajamgam@students.iiit.ac.in).
