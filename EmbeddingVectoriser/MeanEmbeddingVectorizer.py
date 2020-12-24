import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        """
        Parameters
        ----------
        word2vec: Mapping of word to embedding vector.
        """
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X: Documents with tokenised text.

        Returns: Documents with mean weighted word embeddings.
        -------

        """
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
