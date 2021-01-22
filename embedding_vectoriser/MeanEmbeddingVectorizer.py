import numpy as np
from tqdm import tqdm
from time import sleep
from ast import literal_eval
from sklearn.base import TransformerMixin

class MeanEmbeddingVectorizer(TransformerMixin):
    def __init__(self, model, model_type=None, string_input=True):
        self.model = model
        self.vector_dims = model.vector_size
        self.model_type = model_type
        self.string_input = string_input

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_corpus = []
        progress_bar = tqdm(total=len(X))
        for row in X.itertuples():
            sleep(0.01)
            orig_doc = literal_eval(row[1]) if self.string_input else row[1]
            # filter out unseen words for w2v model
            if self.model_type == 'w2v':
                # Numpy zeros if None/ NaN reviews
                doc = np.mean([self.model[word] for word in orig_doc if word in self.model.vocab] or [np.zeros(self.vector_dims)], axis=0)
            else:
                doc = np.mean([self.model[word] for word in orig_doc] or [np.zeros(self.vector_dims)], axis=0)
            new_corpus.append(doc)
            progress_bar.update(1)
        progress_bar.close()
        return np.vstack(new_corpus)
