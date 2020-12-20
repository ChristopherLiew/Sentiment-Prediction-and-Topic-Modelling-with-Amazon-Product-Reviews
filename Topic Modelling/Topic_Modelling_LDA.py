import pandas as pd
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pprint import pprint
from math import floor
from functools import reduce
from ast import literal_eval
from tqdm import tqdm
from time import sleep
import spacy
from gensim import corpora, models
from gensim.utils import ClippedCorpus
import pyLDAvis.gensim
from Text_Preprocessing import preprocess_text, custom_tokenizer
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

### Load Reviews Data ###
amz = pd.read_csv('./Amazon product reviews dataset/Amazon_product_review_with_sent.csv')

### Clean text ###
nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)
tokenizer = custom_tokenizer(nlp)
nlp.tokenizer = tokenizer
clean_amz = preprocess_text(amz, nlp)
# Save preprocessed text
pd.DataFrame(pd.Series(clean_amz), columns=['reviews']).to_csv(
    './Amazon product reviews dataset/Processed_Review_Data/amazon_full_processed.csv')


### POS tag our reviews (Add to preprocessing pipeline) ###
def filter_pos(text, pos_list):  # Consider ADJ & ADVERBS
    refined_reviews = []
    for review in tqdm(text):
        sleep(0.1)
        review_combined = ' '.join(review)
        rev_nlp = nlp(review_combined)
        rev = [word.text for word in rev_nlp if word.pos_ in pos_list]
        refined_reviews.append(rev)
    return refined_reviews


clean_amz_ref = filter_pos(clean_amz, pos_list=['PROPN', 'NOUN', 'VERB'])
pd.DataFrame(pd.Series(clean_amz_ref), columns=['reviews_pos_filtered']).to_csv(
    './Amazon product reviews dataset/Processed_Review_Data/amazon_full_processed_pos_filtered.csv')


### N grams ###
def create_bigrams(text, min_count=20):
    sleep(0.1)
    bigram = models.Phrases(text, min_count)  # Higher min_words and threshold-> Harder to form bigrams
    bigram_mod = models.phrases.Phraser(
        bigram)  # Creates a memory efficient model of phrases w/o model state (Ok if we do not need to train on new docs)
    return [bigram_mod[doc] for doc in text]  # Transform doc to bigrams


def create_trigrams(bigrams, min_count=5):
    trigram = models.Phrases(bigrams, min_count)
    trigram_mod = models.phrases.Phraser(trigram)
    return [trigram_mod[doc] for doc in bigrams]


amz_bigram = create_bigrams(clean_amz_ref)
amz_trigram = create_trigrams(amz_bigram)

### Split dataset into POS, NEG, NEU ###
# amz_ngrams = pd.concat([pd.DataFrame({'reviews_bigrams': amz_bigram}), pd.DataFrame({'reviews_trigrams': amz_trigram}), amz.sentiment], axis=1)
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

### Build LDA model ###
## LDA model function
# To use c_v and c_uci please provide texts for intrinsic measures (i.e. list of list of strings)
def build_lda_model(corpus, dictionary, num_topics, alpha='auto', beta='auto', compute_coherence=True, coherence='u_mass',
                    save=True,
                    saved_model_dir='/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Topic Models/'):
    # Try running on a corpus subset (maybe 50 ~ 75%) else long training time
    lda_model = models.ldamodel.LdaModel(corpus=corpus,
                                         id2word=dictionary,
                                         num_topics=num_topics,
                                         random_state=42,
                                         chunksize=100,
                                         passes=10,
                                         alpha=alpha,
                                         eta=beta,
                                         per_word_topics=True)
    if save:
        try:
            input_path = input("Please enter model name (suffix should be name.model) : ")
            save_path = os.path.join(saved_model_dir, input_path)
            lda_model.save(save_path)
            print("Model successfully saved at: %s" % save_path)
        except(ValueError, FileNotFoundError):
            print("Invalid save path! Please try again.")

    if compute_coherence:
        try:
            print("Computing LDA model coherence ... ")
            coherence_model_lda = models.CoherenceModel(model=lda_model,
                                                        corpus=corpus,
                                                        dictionary=dictionary,
                                                        coherence=coherence)
            print("Returning model and score")
            return lda_model, coherence_model_lda.get_coherence()
        except(ValueError, FileNotFoundError):
            print("File not found")
    else:
        return lda_model

## Hyper-parameter Tuning (Super slow O^N**3)
# See: https://datascience.stackexchange.com/questions/199/what-does-the-alpha-and-beta-hyperparameters-contribute-to-in-latent-dirichlet-a
# and: https://www.thoughtvector.io/blog/lda-alpha-and-beta-parameters-the-intuition/#:~:text=Here%2C%20alpha%20represents%20document%2Dtopic,they%20consist%20of%20few%20words.
# and: https://stackoverflow.com/questions/50607378/negative-values-evaluate-gensim-lda-with-topic-coherence

def tune_lda_model(corpus, dictionary, hyperparameters, val_set_size=0.75, coherence_metric='u_mass'):
    # Try randomised search
    results = {
        'num_topics': [],
        'alpha': [],
        'beta': [],
        'results': []
    }

    num_docs = len(corpus)
    validation_corpus = ClippedCorpus(corpus, max_docs=floor(num_docs * val_set_size))
    num_combinations = reduce((lambda x, y: x * y), [len(value) for key, value in hyperparameters.items()])

    prog_bar = tqdm(total=num_combinations)
    for topic in hyperparameters['num_topics']:
        for alpha in hyperparameters['alpha']:
            for beta in hyperparameters['beta']:
                _, coherence = build_lda_model(validation_corpus, dictionary, coherence=coherence_metric,
                                               num_topics=topic, alpha=alpha, beta=beta, save=False)
                results['num_topics'].append(topic)
                results['alpha'].append(alpha)
                results['beta'].append(beta)
                results['results'].append(coherence)
                prog_bar.update(1)

    prog_bar.close()
    pd.DataFrame(results).to_csv('lda_tuning_results_%s.csv' % datetime.now(), index=False)
    return pd.DataFrame(results)

## Visualise LDA inferred topics
def viz_lda_model(model, corpus, dictionary):
    pyLDAvis.enable_notebook()
    print("Visualising Topics from LDA ...")
    viz = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    return viz

### Train and Evaluate our LDA models ##
## 1) Positive
# Get baseline coherence score
lda_pos, lda_pos_umass_score = build_lda_model(corpus=corpus_tfidf_pos, dictionary=id2wordPos, num_topics=10, save=False)

# Get baseline topics (Sanity check)
pprint(lda_pos.print_topics())

# Tune and find optimal hyper-params (Super slow 120 iterations at ~45 secs/iter)
params_grid = {
    'num_topics': list(range(3, 10, 2)),
    'alpha': list(np.arange(0.01, 1, 0.3)) + ['symmetric', 'asymmetric'],
    'beta': list(np.arange(0.01, 1, 0.3)) + ['symmetric']
}
pos_results = tune_lda_model(corpus=corpus_tfidf_pos, val_set_size=1.0, dictionary=id2wordPos, hyperparameters=params_grid)
pos_results.sort_values(by='results', ascending=False)
# Optimal hyper-params: num_topics = 3, alpha = 0.01, beta = 0.9099999, results = -2.665578

# Num Topics vs. Coherence Score
sns.lineplot(data=pos_results[['num_topics', 'results']], x="num_topics", y="results")
plt.title('Positive: No. of Topics vs. Coherence')
plt.show()

# Build & Save Optimal Positive Model & Get Topics
lda_opt_pos, score = build_lda_model(corpus_tfidf_pos, id2wordPos, num_topics=3, alpha=0.01, beta=0.9099999999999999)
pprint(lda_opt_pos.print_topics())

# 2) Neutral
neu_results = tune_lda_model(corpus=corpus_tfidf_neu, dictionary=id2wordNeu, hyperparameters=params_grid)
neu_results.sort_values(by='results', ascending=False)
# Optimal hyper-params: num_topics = 3, alpha = 0.01, beta = 0.01, results = -4.190391

# Num Topics vs. Coherence Score
sns.lineplot(data=neu_results[['num_topics', 'results']], x="num_topics", y="results")
plt.title('Neutral: No. of Topics vs. Coherence')
plt.show()

# Build & Save Optimal Neutral Model & Get Topics
lda_opt_neu, score_neu = build_lda_model(corpus_tfidf_neu, id2wordNeu, num_topics=3, alpha=0.01, beta=0.01)
pprint(lda_opt_neu.print_topics())

# 3) Negative
neg_results = tune_lda_model(corpus=corpus_tfidf_neg, dictionary=id2wordNeg, hyperparameters=params_grid)
neg_results.sort_values(by='results', ascending=False)
# Optimal hyper-params: num_topics = 9, alpha = 0.91, beta = 0.91, results = -2.593273

# Num Topics vs. Coherence Score
sns.lineplot(data=neg_results[['num_topics', 'results']], x="num_topics", y="results")
plt.title('Negative: No. of Topics vs. Coherence')
plt.show()

# Build & Save Optimal Negative Model & Get Topics
lda_opt_neg, score_neg = build_lda_model(corpus_tfidf_neg, id2wordNeg, num_topics=9, alpha=0.91, beta=0.91)
pprint(lda_opt_neg.print_topics())