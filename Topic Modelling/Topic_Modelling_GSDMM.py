import pandas as pd
import pprint
import pickle
import tqdm
from nltk import ngrams
import spacy
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
from Text_Preprocessing import preprocess_text, custom_tokenizer
from gsdmm import MovieGroupProcess

# Load amazon data with sentiments
amz = pd.read_csv('./Amazon product reviews dataset/Amazon_product_review_with_sent.csv')

### Clean text ###
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
tokenizer = custom_tokenizer(nlp)
nlp.tokenizer = tokenizer
clean_amz = preprocess_text(amz, nlp)

### POS tag our reviews ###
def get_topic_pos(text, pos_list=['PROPN', 'NOUN', 'VERB']): # Consider ADJ & ADVERBS
    refined_reviews = []
    for review in text:
        review_combined = ' '.join(review)
        rev_nlp = nlp(review_combined)
        rev = [word.text for word in rev_nlp if word.pos_ in pos_list]
        refined_reviews.append(rev)
    return refined_reviews

clean_amz_ref = get_topic_pos(clean_amz)

### N grams ### 
# Get bigrams & trigrams (NLTK)
def construct_ngrams(text, ngram=2):
    ngram_reviews = []
    for review in text:
        ngram_reviews.append([i for i in ngrams(review, ngram)])
    return ngram_reviews

# Bigram model (Gensim)
def create_bigrams(text): 
    bigram = models.Phrases(text)  # Higher min_words and threshold-> Harder to form bigrams
    bigram_mod = models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in text]

amz_bigram = create_bigrams(clean_amz_ref)

### Split dataset into POS, NEG, NEU ###
amz_bigram_with_sent = pd.concat([pd.DataFrame({'reviews.bigrams': amz_bigram}), amz.sentiment], axis=1)
amz_bigram_pos = list(amz_bigram_with_sent[amz_bigram_with_sent['sentiment'] == 'Positive']['reviews.bigrams'])
amz_bigram_neu = list(amz_bigram_with_sent[amz_bigram_with_sent['sentiment'] == 'Neutral']['reviews.bigrams'])
amz_bigram_neg = list(amz_bigram_with_sent[amz_bigram_with_sent['sentiment'] == 'Negative']['reviews.bigrams'])

### LDA model ###
# Create Dictionary 
id2wordPos = corpora.Dictionary(amz_bigram_pos)
id2wordNeu = corpora.Dictionary(amz_bigram_neu)
id2wordNeg = corpora.Dictionary(amz_bigram_neg)

# Create Corpus
texts_pos = amz_bigram_pos
texts_neu = amz_bigram_neu
texts_neg = amz_bigram_neg

# Term Document Frequency (For each doc we have word_id: no. of times it occurs within that document)
corpus_pos = [id2wordPos.doc2bow(text) for text in texts_pos]
corpus_neu = [id2wordNeu.doc2bow(text) for text in texts_neu]
corpus_neg = [id2wordNeg.doc2bow(text) for text in texts_neg]

# TF-IDF

# Neural Word Embeddings (TBD)

# For readability
# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:7]]

# Build LDA model
# POS LDA
lda_model_pos = models.ldamodel.LdaModel(corpus=corpus_pos,
                                         id2word=id2wordPos,
                                         num_topics=20,
                                         random_state=42,
                                         update_every=1,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         eta='auto',
                                         per_word_topics=True)

# Get Positive Review Topics
pprint.pprint(lda_model_pos.print_topicarpams())

# Visualize the topics related to Positive Reviews
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_pos, corpus_pos, id2wordPos)
vis

# Get coherence score
coherence_model_lda_pos = models.CoherenceModel(model=lda_model_pos, texts=texts_pos, dictionary=id2wordPos, coherence='c_v')
coherence_lda_pos = coherence_model_lda_pos.get_coherence()
print('Coherence Score: ', coherence_lda_pos)

# NEU LDA
lda_model_neu = models.ldamodel.LdaModel(corpus=corpus_neu,
                                         id2word=id2wordNeu,
                                         num_topics=20,
                                         random_state=42,
                                         update_every=1,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         eta='auto',
                                         per_word_topics=True)

# Get topics
pprint.pprint(lgda_model_neu.print_topics())
# Visualize the topics related to Neutral Reviews
pyLDAvis.enable_notebook()
vis_neu = pyLDAvis.gensim.prepare(lda_model_neu, corpus_neu, id2wordNeu)
vis_neu

# NEG LDA
lda_model_neg = models.ldamodel.LdaModel(corpus=corpus_neg,
                                         id2word=id2wordNeg,
                                         num_topics=20,
                                         random_state=42,
                                         update_every=1,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         eta='auto',
                                         per_word_topics=True)

# Get topics
pprint.pprint(lda_model_neg.print_topics())
# Visualize the topics related to Neutral Reviews
pyLDAvis.enable_notebook()
vis_neg = pyLDAvis.gensim.prepare(lda_model_neg, corpus_neg, id2wordNeg)
vis_neg
