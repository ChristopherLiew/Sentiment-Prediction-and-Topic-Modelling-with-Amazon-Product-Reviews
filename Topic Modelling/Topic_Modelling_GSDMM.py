import pandas as pd
import numpy as np
from ast import literal_eval
from gsdmm import MovieGroupProcess
from gensim import corpora, models

### Split dataset into POS, NEG, NEU ###
# Load existing ngram data from Topic_Modelling_LDA.py file:

amz_ngrams = pd.read_csv(
    '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Amazon product reviews dataset/Processed_Review_Data/amazon_full_ngrams.csv',
    index_col=False)

amz_ngrams_pos = list(amz_ngrams[amz_ngrams['sentiment'] == 'Positive']['reviews_trigrams'])
amz_ngrams_neu = list(amz_ngrams[amz_ngrams['sentiment'] == 'Neutral']['reviews_trigrams'])
amz_ngrams_neg = list(amz_ngrams[amz_ngrams['sentiment'] == 'Negative']['reviews_trigrams'])


def convert_to_lists(string_data):
    '''
    Use when converting back from csv data.
    '''
    return [literal_eval(i) for i in string_data]


amz_ngrams_pos = convert_to_lists(amz_ngrams_pos)
amz_ngrams_neu = convert_to_lists(amz_ngrams_neu)
amz_ngrams_neg = convert_to_lists(amz_ngrams_neg)

### Construct TF-IDF corpus ###
# Create Dictionary & Filter very rare words
id2wordPos = corpora.Dictionary(amz_ngrams_pos)
id2wordPos.filter_extremes(no_below=20, no_above=0.5)  # Extra layer of filtering on top of stop words etc

id2wordNeu = corpora.Dictionary(amz_ngrams_neu)
id2wordNeu.filter_extremes(no_below=20, no_above=0.5)

id2wordNeg = corpora.Dictionary(amz_ngrams_neg)
id2wordNeg.filter_extremes(no_below=20, no_above=0.5)

# TF-IDF
# Convert to Bag of Words using Dictionary
corpus_pos = [id2wordPos.doc2bow(text) for text in amz_ngrams_pos]
corpus_neu = [id2wordNeu.doc2bow(text) for text in amz_ngrams_neu]
corpus_neg = [id2wordNeg.doc2bow(text) for text in amz_ngrams_neg]

# Convert to TF-IDF from BOW
tfidf_pos = models.TfidfModel(corpus_pos)  # construct TF-IDF model to convert any BOW rep to TF-IDF rep
corpus_tfidf_pos = tfidf_pos[corpus_pos]  # Convert corpus to TF-IDF rep

tfidf_neu = models.TfidfModel(corpus_neu)
corpus_tfidf_neu = tfidf_pos[corpus_neu]

tfidf_neg = models.TfidfModel(corpus_neg)
corpus_tfidf_neg = tfidf_pos[corpus_neg]

### Build GSDMM model aka Movie Group Process ###
## Positive Corpus
mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=30)
topics = mgp.fit(corpus_tfidf_pos, len(id2wordPos))

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sorted_dicts = sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print('Cluster %s : %s' % (cluster, sorted_dicts))

doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# Topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[-10:][::-1]
print('Most important clusters (by number of docs inside):', top_index)

# Show the top 10 words in term frequency for each cluster
top_words(mgp.cluster_word_distribution, top_index, 10)
