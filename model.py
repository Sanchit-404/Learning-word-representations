# model.py

from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize

class WordEmbeddingModel:
    def __init__(self, corpus, embedding_type='word2vec', output_model_file='word_embedding_model'):
        self.corpus = corpus
        self.embedding_type = embedding_type
        self.output_model_file = output_model_file
        self.model = self.train_model()

    def train_model(self):
        model_types = {
            'word2vec': Word2Vec,
            'fasttext': FastText,
            'glove': self._train_glove_like_model,
        }

        model_class = model_types.get(self.embedding_type.lower())

        if model_class is None:
            raise ValueError("Invalid embedding type. Supported options: 'word2vec', 'fasttext', 'glove'")

        if self.embedding_type.lower() != 'glove':
            model = model_class(sentences=[self.corpus], vector_size=100, window=5, min_count=1, workers=4)
        else:
            model = model_class(self.corpus, self.output_model_file)

        model.save(self.output_model_file)
        return model

    @staticmethod
    def _train_glove_like_model(corpus, output_model_file):
        tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
        model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=10, sg=0, min_count=1, workers=4)
        model.save(output_model_file)
    def get_word_vector(self, word):
        if self.embedding_type.lower() == 'word2vec':
            try:
                return self.model.wv[word]
            except KeyError:
                return None
        elif self.embedding_type.lower() == 'fasttext':
            # For FastText, use the following syntax to get word vectors
            try:
                return self.model.wv.get_vector(word)
            except KeyError:
                return None
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")

