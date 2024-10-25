# Semantically Similar Code Snippet Retrieval (SSCSR)

## Project Overview

The SSCSR project is tasked with retrieving code snippets that are semantically similar to a given code query using the POJ-104 dataset. This stage involves preprocessing of the data to ensure it is formatted correctly for subsequent feature extraction and model training.

## Dataset

We utilize the [POJ-104](https://arxiv.org/pdf/1409.5718.pdf) dataset, which contains pairs of semantically similar code snippets, for this task.

### Download and Preprocess

1. **Download dataset** from [website](https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?usp=sharing) or run the following command:
   ```shell
   cd dataset
   pip install gdown
   gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
   tar -xvf programs.tar.gz
   cd ..
   ```

2. **Preprocess data** with the script prepared by Yuvaraj Gajalajamjam:
   ```shell
   cd dataset
   python preprocess.py
   cd ..
   ```

### Data Format

After preprocessing, you will have three .jsonl files: train.jsonl, valid.jsonl, and test.jsonl. Each line in these files represents one function, formatted as follows:
- **code:** the source code.
- **label:** the problem number that the code snippet solves.
- **index:** the example index.

### Data Statistics

Here are the statistics of the dataset after preprocessing:

|       | #Problems | #Examples |
| ----- | --------- | :-------: |
| Train | 64        |  32,000   |
| Dev   | 16        |   8,000   |
| Test  | 24        |  12,000   |

## Current Stage: Data Preprocessing

This stage's key contributions include:
- **Normalization and Tokenization**: Converting the raw code snippets into a standardized format.
- **Splitting Data**: Dividing the dataset into training, validation, and testing sets to prepare for the model training phase.

### Preprocessing Script

Yuvaraj Gajalajamjam has prepared the preprocessing script, which normalizes and tokenizes the data, preparing it for the next stages of the project.

## Next Steps

- **Feature Extraction**: Techniques such as Word2Vec will be used for vectorizing code tokens.
- **Semantic Embedding**: Code vectors will be transformed into semantic embeddings using models like CodeBERT.

## Repository Structure

```
/preprocessing/ - Contains Jupyter notebooks and scripts for data preprocessing.
```

## Installation and Usage

To set up the project:
1. **Clone the repository**:
   ```
   !git clone https://github.com/microsoft/CodeXGLUE.git
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Run the preprocessing script**:
   ```
   python preprocessing/pre_processing_similar_code_snippets.ipynb
   ```

Ensure to replace `<repo-url>` with the actual URL of your GitHub repository. This README provides a complete overview of the preprocessing stage, setting the stage for subsequent phases of the project.