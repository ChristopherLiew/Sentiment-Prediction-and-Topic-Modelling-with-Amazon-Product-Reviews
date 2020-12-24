import pandas as pd
import gensim
import logging
from ast import literal_eval
from tqdm import tqdm
from time import sleep
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# TBD:
# 1) Test classifiers with w2v and fast_text embeddings

####################################################
#  Evaluation metrics: macro-avg F1 + accuracy     #
####################################################

# Load training and testing data
amz_aug_train_data = pd.read_csv('./Amazon product reviews dataset/Synonym_augmented_data/amazon_syn_aug_train_processed.csv')
amz_train_data = pd.read_csv('./Amazon product reviews dataset/amazon_train_processed.csv')
amz_test_data = pd.read_csv('./Amazon product reviews dataset/amazon_test_processed.csv')
amz_combined_data = pd.concat([amz_train_data, amz_test_data], axis=0)

## Text Preprocessing
## A. TF-IDF
def identity_tokenizer(text):
    return text

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)
amz_aug_train_data_tfidf = tfidf.fit_transform(amz_aug_train_data.text_processed)
amz_train_data_tfidf = tfidf.fit_transform(amz_train_data.text_processed)
amz_test_data_tfidf = tfidf.fit_transform(amz_test_data.text_processed)

####################################################
#  Word Embeddings: Word 2 Vec & Fast Text         #
####################################################

## B. Word2Vec (KIV - Training W2V model, unlikely to better google's pre-trained vectors)
# Load Google's pre-trained 300D word2vec model
w2v = gensim.models.KeyedVectors.load_word2vec_format('/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/nlpaug_model_dir/GoogleNews-vectors-negative300.bin', binary=True)

def convert_to_w2v(corpus, w2v_vectors, convert_to_string=False, is_w2v=True):
    new_corpus = []
    prog_bar = tqdm(total=len(corpus))
    for row in corpus.itertuples():
        sleep(0.01)
        orig_doc = literal_eval(row[1])
        # filter out unseen words
        if is_w2v:
            orig_doc = filter(lambda x: x in w2v_vectors.vocab, orig_doc)
        if convert_to_string:
            w2v_doc = str([w2v_vectors[word] for word in orig_doc])
        else:
            w2v_doc = [w2v_vectors[word] for word in orig_doc]
        new_corpus.append(w2v_doc)
        prog_bar.update(1)
    prog_bar.close()
    return new_corpus

# Training Data
w2v_amz_train_data = convert_to_w2v(amz_train_data, w2v, convert_to_string=True)
w2v_amz_train_data = pd.concat([pd.DataFrame(w2v_amz_train_data, columns=['reviews']), amz_train_data.sentiment], axis=1)

# Testing Data
w2v_amz_test_data = convert_to_w2v(amz_test_data, w2v, convert_to_string=True)
w2v_amz_test_data = pd.concat([pd.DataFrame(w2v_amz_test_data, columns=['reviews']), amz_test_data.sentiment], axis=1)

# Save training/ testing data
w2v_amz_train_data.to_csv('/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Amazon product reviews dataset/Processed_Review_Data/w2v_amz_train_data.csv', index=False)
w2v_amz_test_data.to_csv('/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Amazon product reviews dataset/Processed_Review_Data/w2v_amz_test_data.csv', index=False)

## C. FastText Embeddings - Handles out of vocab vectors using composite n_grams
# 1) Create un-tokenised text data
tqdm.pandas(desc="Progress bar")
amz_combined_data_unified = pd.DataFrame(amz_combined_data.progress_apply(lambda x: str(' '.join(literal_eval(x['text_processed']))), axis=1), columns=['text_processed'])

# 2) train and save Fast Text model
fast_text_model = FastText(window=5, min_count=5)
# Build vocab
docs = amz_combined_data_unified['text_processed'].to_list()
fast_text_model.build_vocab(sentences=docs)
# Train model
# train the model
fast_text_model.train(
    sentences=docs,
    epochs=fast_text_model.epochs,
    total_examples=fast_text_model.corpus_count,
    total_words=fast_text_model.corpus_total_words
)

fast_text_model.save('/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Word Embedding Models/fast_text.model')

# 3) Created new reviews data using embeddings
fast_text_corpus = convert_to_w2v(amz_train_data, fast_text_model, convert_to_string=True, is_w2v=False)
fast_text_corpus = pd.concat([pd.DataFrame(fast_text_corpus, columns=['reviews']), amz_train_data.sentiment], axis=1)
fast_text_corpus.to_csv('/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Amazon product reviews dataset/Processed_Review_Data/fast_text_amz_data.csv', index=False)

##########################################
#       Building Features from Text      #
##########################################
# Mean Embedding Vectoriser
# TF-IDF Embedding Vectoriser

##########################################
#        Model Development & Eval        #
##########################################
# Auxiliary Function for Results
def get_clf_results(y_true, y_pred):
    print(multilabel_confusion_matrix(y_true, y_pred))
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))

## 1. Multinomial Naive Bayes classifier 
# Augmented Data
gnb_clf_aug = MultinomialNB()
gnb_clf_aug.fit(amz_aug_train_data_tfidf.toarray(),amz_aug_train_data.sentiment)
amz_aug_pred_nb = gnb_clf_aug.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_aug_pred_nb)

# Unaugmented Data
gnb_clf = MultinomialNB()
gnb_clf.fit(amz_train_data_tfidf.toarray(), amz_train_data.sentiment)
amz_pred_nb = gnb_clf.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_pred_nb)

## 2. Support Vector Classifier 

# Hyperparameter Tuning
params_dict = {
    'kernel': ['linear'],
    'C': [0.5, 0.75, 1.0],
    'gamma': ['auto'],
    'class_weight': ['balanced']
}

svc_clf_tuned = RandomizedSearchCV(SVC(), params_dict, random_state=42)

# Augmented
search_augmented = svc_clf_tuned.fit(amz_aug_train_data_tfidf.toarray(), amz_aug_train_data.sentiment)
search_augmented.best_score_
search_augmented.best_params_

# Unaugmented
search_norm = svc_clf_tuned.fit(amz_train_data_tfidf.toarray(), amz_train_data.sentiment)
search_norm.best_score_
search_norm.best_params_

# Augmented Data
svc_clf = SVC(kernel='linear', gamma='auto', class_weight='balanced')
svc_clf.fit(amz_aug_train_data_tfidf.toarray(), amz_aug_train_data.sentiment)
amz_aug_pred_svc = svc_clf.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_aug_pred_svc)

# Unaugmented Data
svc_clf.fit(amz_train_data_tfidf.toarray(), amz_train_data.sentiment)
amz_pred_svc = svc_clf.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_pred_svc)

## 3. Random Forest
params_dict_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'class_weight': ['balanced']
}

rf_clf_tuned = RandomizedSearchCV(RandomForestClassifier(), params_dict_rf, random_state=42)
# Augmented
search_augmented_rf = rf_clf_tuned.fit(amz_aug_train_data_tfidf.toarray(), amz_aug_train_data.sentiment)
search_augmented_rf.best_score_
search_augmented_rf.best_params_

# Unaugmented
search_norm_rf = rf_clf_tuned.fit(amz_train_data_tfidf.toarray(), amz_train_data.sentiment)
search_norm_rf.best_score_
search_norm_rf.best_params_

# Augmented Data (Optimal hyperparams)
rf_clf_aug = RandomForestClassifier(n_estimators=300, max_depth=4, class_weight='balanced', oob_score=True)
rf_clf_aug.fit(amz_aug_train_data_tfidf.toarray(),amz_aug_train_data.sentiment)
amz_aug_pred_rf = rf_clf_aug.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_aug_pred_rf)

# Unaugmented Data 
rf_clf_unaug = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', oob_score=True)
rf_clf_unaug.fit(amz_train_data_tfidf.toarray(), amz_train_data.sentiment)
amz_pred_rf = rf_clf_unaug.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_pred_rf)
