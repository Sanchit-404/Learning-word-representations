from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import jaccard_score
from scipy.stats import pearsonr, spearmanr

class SimilarityMetrics:
    def __init__(self, vector_col1, vector_col2):
        self.vector_col1 = vector_col1
        self.vector_col2 = vector_col2
        self.metric_functions = {
            'cosine_similarity': self._cosine_similarity,
            'euclidean_distance': self._euclidean_distance,
            'manhattan_distance': self._manhattan_distance,
            'jaccard_similarity': self._jaccard_similarity,
            'pearson_similarity': self._pearson_similarity,
            'spearman_similarity': self._spearman_similarity,
        }

    @staticmethod
    def _cosine_similarity(vector1, vector2):
        return cosine_similarity([vector1], [vector2])[0][0]

    @staticmethod
    def _euclidean_distance(vector1, vector2):
        return 1 / (1 + pairwise_distances([vector1], [vector2], metric='euclidean')[0][0])

    @staticmethod
    def _manhattan_distance(vector1, vector2):
        return 1 / (1 + pairwise_distances([vector1], [vector2], metric='manhattan')[0][0])

    @staticmethod
    def _jaccard_similarity(vector1, vector2):
        return jaccard_score(vector1.astype(bool).flatten(), vector2.astype(bool).flatten())

    @staticmethod
    def _pearson_similarity(vector1, vector2):
        correlation_coefficient = pearsonr(vector1.flatten(), vector2.flatten())[0]
        return 0.5 + 0.5 * correlation_coefficient

    @staticmethod
    def _spearman_similarity(vector1, vector2):
        rank_correlation_coefficient = spearmanr(vector1.flatten(), vector2.flatten())[0]
        return 0.5 + 0.5 * rank_correlation_coefficient

    def compute_similarity(self, metric_name, row):
        vector1 = row[self.vector_col1]
        vector2 = row[self.vector_col2]

        if metric_name in self.metric_functions:
            return self.metric_functions[metric_name](vector1, vector2)
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")
