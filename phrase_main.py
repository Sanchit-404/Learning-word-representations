# main.py

import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def train_word2vec_model(sentences):
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None

def main():
    # Load the PiC dataset from hf
    pic_dataset = load_dataset('PiC/phrase_similarity', 'PS-hard')

    # Convert the Hugging Face dataset to a df
    pic_df = pd.DataFrame(pic_dataset['train'])

    # Train a Word2Vec model on corpus.txt
    corpus = pic_df['sent1'].tolist() + pic_df['sent2'].tolist()
    word2vec_model = train_word2vec_model([sentence.split() for sentence in corpus])

    # create embeddings for sent1 and sent2
    pic_df['vec_sent1'] = pic_df['sent1'].apply(lambda x: get_sentence_vector(x, word2vec_model))
    pic_df['vec_sent2'] = pic_df['sent2'].apply(lambda x: get_sentence_vector(x, word2vec_model))
    pic_df = pic_df.dropna(subset=['vec_sent1', 'vec_sent2'])

    # Calculate cosine similarity
    pic_df['cosine_similarity'] = pic_df.apply(lambda row: cosine_similarity([row['vec_sent1']], [row['vec_sent2']])[0][0], axis=1)
    X = pic_df[['cosine_similarity']]
    y = pic_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions 
    y_pred = model.predict(X_test)

    # Evaluate 
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
