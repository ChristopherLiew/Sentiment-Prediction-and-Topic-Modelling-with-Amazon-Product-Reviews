import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        """
        Parameters
        ----------
        word2vec: Mapping of word to embedding vector.
        """
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: Documents with tokenised text.
        y: Target feature.

        Returns: Fitted TfidfEmbeddingVectoriser object.
        -------

        """
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X: Documents with tokenised text.

        Returns: Documents with tfidf weighted word embeddings.
        -------

        """
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])
