import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pprint import pprint
from math import floor
from functools import reduce
from ast import literal_eval
from tqdm import tqdm
from gensim import corpora, models
from gensim.utils import ClippedCorpus
import pyLDAvis.gensim
import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

### Split dataset into POS, NEG, NEU ###
# Load existing ngram data from Topic_Modelling_LDA.py file:

amz_ngrams = pd.read_csv(
    '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Amazon product reviews dataset/Processed_Review_Data/amazon_full_ngrams.csv',
    index_col=False)

amz_ngrams_pos = list(amz_ngrams[amz_ngrams['sentiment'] == 'Positive']['reviews_trigrams'])
amz_ngrams_neu = list(amz_ngrams[amz_ngrams['sentiment'] == 'Neutral']['reviews_trigrams'])
amz_ngrams_neg = list(amz_ngrams[amz_ngrams['sentiment'] == 'Negative']['reviews_trigrams'])


# amz_ngrams.to_csv('./Amazon product reviews dataset/Processed_Review_Data/amazon_full_ngrams.csv', index=False)

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


### Build LSI model ###
## LSI model function
# To use c_v and c_uci please provide texts for intrinsic measures (i.e. list of list of strings)
def build_lsi_model(corpus, dictionary, num_topics, compute_coherence=True, coherence='u_mass',
                    save=True,
                    saved_model_dir='/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Topic Models/LSI_models/'):
    # Try running on a corpus subset (maybe 50 ~ 75%) else long training time
    lsi_model = models.lsimodel.LsiModel(corpus=corpus,
                                         id2word=dictionary,
                                         num_topics=num_topics,
                                         chunksize=100)
    if save:
        try:
            input_path = input("Please enter model name (suffix should be name.model) : ")
            save_path = os.path.join(saved_model_dir, input_path)
            lsi_model.save(save_path)
            print("Model successfully saved at: %s" % save_path)
        except(ValueError, FileNotFoundError):
            print("Invalid save path! Please try again.")

    if compute_coherence:
        try:
            print("Computing LDA model coherence ... ")
            coherence_model_lsi = models.CoherenceModel(model=lsi_model,
                                                        corpus=corpus,
                                                        dictionary=dictionary,
                                                        coherence=coherence)
            print("Returning model and score")
            return lsi_model, coherence_model_lsi.get_coherence()
        except(ValueError, FileNotFoundError):
            print("File not found")
    else:
        return lsi_model


## Hyper-parameter Tuning (Super slow O^N**3)
# See: https://datascience.stackexchange.com/questions/199/what-does-the-alpha-and-beta-hyperparameters-contribute-to-in-latent-dirichlet-a
# and: https://www.thoughtvector.io/blog/lda-alpha-and-beta-parameters-the-intuition/#:~:text=Here%2C%20alpha%20represents%20document%2Dtopic,they%20consist%20of%20few%20words.
# and: https://stackoverflow.com/questions/50607378/negative-values-evaluate-gensim-lda-with-topic-coherence

def tune_lsi_model(corpus, dictionary, hyperparameters, val_set_size=0.75, coherence_metric='u_mass'):
    # Try randomised search
    results = {
        'num_topics': [],
        'results': []
    }

    num_docs = len(corpus)
    validation_corpus = ClippedCorpus(corpus, max_docs=floor(num_docs * val_set_size))
    num_combinations = reduce((lambda x, y: x * y), [len(value) for key, value in hyperparameters.items()])

    prog_bar = tqdm(total=num_combinations)
    for topic in hyperparameters['num_topics']:
        _, coherence = build_lsi_model(validation_corpus, dictionary, coherence=coherence_metric,
                                       num_topics=topic, save=False)
        results['num_topics'].append(topic)
        results['results'].append(coherence)
        prog_bar.update(1)

    prog_bar.close()
    return pd.DataFrame(results)

### Train and Evaluate our LSI models ##
## 1) Positive
# Get baseline coherence score
lsi_pos, lsi_pos_umass_score = build_lsi_model(corpus=corpus_tfidf_pos, dictionary=id2wordPos, num_topics=10, save=False)
# Baseline score: -4.0257787996201655
# Sanity Check Topics
pprint(lsi_pos.print_topics())

# Tune and find optimal hyper-params
params_grid = {
    'num_topics': list(range(3, 15, 2))
}
pos_results = tune_lsi_model(corpus=corpus_tfidf_pos, val_set_size=1.0, dictionary=id2wordPos, hyperparameters=params_grid)
pos_results.sort_values(by='results', ascending=False)
# Optimal hyper-params: num_topics = 5, results = -3.369051

# Num Topics vs. Coherence Score
sns.lineplot(data=pos_results[['num_topics', 'results']], x="num_topics", y="results")
plt.title('Positive: No. of Topics vs. Coherence')
plt.show()

# Build & Save Optimal Positive Model & Get Topics
lsi_opt_pos, score = build_lsi_model(corpus_tfidf_pos, id2wordPos, num_topics=5)
pprint(lsi_opt_pos.print_topics())

# 2) Neutral
# Get baseline coherence score
lsi_neu, lsi_neu_umass_score = build_lsi_model(corpus=corpus_tfidf_neu, dictionary=id2wordNeu, num_topics=10, save=False)
# Baseline score: -6.377077933106273
# Sanity Check Topics
pprint(lsi_neu.print_topics())

params_grid = {
    'num_topics': list(range(2, 15, 1))
}

neu_results = tune_lsi_model(corpus=corpus_tfidf_neu, dictionary=id2wordNeu, hyperparameters=params_grid)
neu_results.sort_values(by='results', ascending=False)
# Optimal hyper-params: num_topics = 12, results = -7.088210

# Num Topics vs. Coherence Score
sns.lineplot(data=neu_results[['num_topics', 'results']], x="num_topics", y="results")
plt.title('Neutral: No. of Topics vs. Coherence')
plt.show()

# Build & Save Optimal Neutral Model & Get Topics
lsi_opt_neu, score_neu = build_lsi_model(corpus_tfidf_neu, id2wordNeu, num_topics=12)
pprint(lsi_opt_neu.print_topics())

# 3) Negative
# Get baseline coherence score
lsi_neg, lsi_neg_umass_score = build_lsi_model(corpus=corpus_tfidf_neg, dictionary=id2wordNeg, num_topics=10, save=False)
# Baseline score: -4.504442474828392
# Sanity Check Topics
pprint(lsi_neg.print_topics()) # More informative

neg_results = tune_lsi_model(corpus=corpus_tfidf_neg, dictionary=id2wordNeg, hyperparameters=params_grid)
neg_results.sort_values(by='results', ascending=False)
# Optimal hyper-params: num_topics = 2, results = -2.298504

# Num Topics vs. Coherence Score
sns.lineplot(data=neg_results[['num_topics', 'results']], x="num_topics", y="results")
plt.title('Negative: No. of Topics vs. Coherence')
plt.show()

# Build & Save Optimal Negative Model & Get Topics
lsi_opt_neg, score_neg = build_lsi_model(corpus_tfidf_neg, id2wordNeg, num_topics=2)
pprint(lsi_opt_neg.print_topics())
