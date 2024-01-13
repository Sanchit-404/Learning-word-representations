import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from datasets import load_dataset

dataset = load_dataset("paws", "labeled_final")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# tokenize and get BERT embeddings
def get_bert_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    out = model(**inputs)
    embeddings = out.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# calculate cosine similarity
def calc_similarity(embedding1, embedding2):
    return np.squeeze(cosine_similarity(embedding1, embedding2))

# Generate embeddings 
train_embeddings = [
    calc_similarity(
        get_bert_embedding(item["sentence1"], tokenizer, model),
        get_bert_embedding(item["sentence2"], tokenizer, model)
    ) for item in train_data
]
val_embeddings = [
    calc_similarity(
        get_bert_embedding(item["sentence1"], tokenizer, model),
        get_bert_embedding(item["sentence2"], tokenizer, model)
    ) for item in val_data
]

# Train classifier
classifier = LogisticRegression()
classifier.fit(np.array(train_embeddings).reshape(-1, 1), [item["label"] for item in train_data])

# Inference
val_predictions = classifier.predict(np.array(val_embeddings).reshape(-1, 1))
val_accuracy = accuracy_score([item["label"] for item in val_data], val_predictions)
print(f'Validation Accuracy (BERT): {val_accuracy}')

# Generate embeddings for test
test_embeddings = [
    calc_similarity(
        get_bert_embedding(item["sentence1"], tokenizer, model),
        get_bert_embedding(item["sentence2"], tokenizer, model)
    ) for item in test_data
]

# Predict on test
test_predictions = classifier.predict(np.array(test_embeddings).reshape(-1, 1))

test_accuracy = accuracy_score([item["label"] for item in test_data], test_predictions)
print(f'Test Accuracy (BERT): {test_accuracy}')
