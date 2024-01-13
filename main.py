# main.py

import argparse
from load_dataset import DatasetLoader
from model import WordEmbeddingModel
from metrics import SimilarityMetrics
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Word Embedding and Similarity Metrics Evaluation')
    parser.add_argument('--simlex_path', type=str, default=r'C:\Users\sanch\Downloads\SimLex-999\SimLex-999\SimLex-999.txt', help='Path to SimLex-999 dataset')
    parser.add_argument('--embedding_type', type=str, default='fasttext', choices=['word2vec', 'fasttext', 'glove'], help='Embedding type (word2vec, fasttext, glove)')
    parser.add_argument('--metric_name', type=str, default='cosine_similarity', choices=['cosine', 'euclidean', 'manhattan', 'jaccard', 'pearson', 'spearman'], help='Similarity metric name')

    args = parser.parse_args()

    # Load Simlex dataset
    loader = DatasetLoader(target_tokens=100000)
    simlex_df = loader.load_simlex_dataset(args.simlex_path)

    # Load or train word wmbedding model
    corpus_file = 'corpus.txt'  # Default name, can be changed as per need
    with open(corpus_file, 'r', encoding='utf-8') as file:
        corpus = file.read().split()

    output_model_file = 'word_embedding_model'
    word_embedding_model = WordEmbeddingModel(corpus, args.embedding_type, output_model_file)
    trained_model = word_embedding_model.train_model()

    # Get vectors 
    simlex_df['vector_word1'] = simlex_df['word1'].apply(word_embedding_model.get_word_vector)
    simlex_df['vector_word2'] = simlex_df['word2'].apply(word_embedding_model.get_word_vector)
    #Calculate similarity based on choice
    similarity_metrics = SimilarityMetrics(vector_col1='vector_word1', vector_col2='vector_word2')

    # Create a metric column
    if args.metric_name not in simlex_df.columns:
        simlex_df[args.metric_name] = np.nan

    # Calculate the specified metric for df
    simlex_df[args.metric_name] = simlex_df.apply(lambda row: similarity_metrics.compute_similarity(args.metric_name, row), axis=1)

    # Evaluate similarity predictions
    mse, mae, pearson_corr, spearman_corr = evaluate_similarity_predictions(simlex_df['SimLex999'], simlex_df[args.metric_name], args.metric_name)

    # Display results
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'Pearson Correlation: {pearson_corr}')
    print(f'Spearman Correlation: {spearman_corr}')

    # Visualize normalized scores
    evaluate_similarity_predictions(simlex_df['SimLex999'], simlex_df[args.metric_name], args.metric_name)

def evaluate_similarity_predictions(actual, predicted, metric_name):
    # Normalize scores to be in the range [0, 1]
    scaler = MinMaxScaler()
    actual_normalized = scaler.fit_transform(actual.values.reshape(-1, 1))
    predicted_normalized = scaler.fit_transform(predicted.values.reshape(-1, 1))

    actual_normalized = pd.Series(actual_normalized.flatten())
    predicted_normalized = pd.Series(predicted_normalized.flatten())

    # Calculate correlation coefficients
    pearson_corr = actual_normalized.corr(predicted_normalized)
    spearman_corr = actual_normalized.corr(predicted_normalized, method='spearman')

    # Visualize results
    plt.scatter(actual_normalized, predicted_normalized)
    plt.xlabel(f'{metric_name} Normalized Ground Truth')
    plt.ylabel('Predicted Normalized Score')
    plt.title('Normalized Scores Comparison')
    plt.show()

    # Evaluate performance metrics
    mse = mean_squared_error(actual_normalized, predicted_normalized)
    mae = mean_absolute_error(actual_normalized, predicted_normalized)

    return mse, mae, pearson_corr, spearman_corr

if __name__ == "__main__":
    main()
