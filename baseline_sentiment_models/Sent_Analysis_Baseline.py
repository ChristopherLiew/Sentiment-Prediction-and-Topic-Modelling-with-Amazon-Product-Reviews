import pandas as pd
import gensim
import logging
import pickle
import gensim.downloader
from ast import literal_eval
from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from embedding_vectoriser.MeanEmbeddingVectorizer import MeanEmbeddingVectorizer
from embedding_vectoriser.TfidfEmbeddingVectorizer import TfidfEmbeddingVectorizer

# Formatting
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10000)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

### TBD ###
# 1) Plot learning & Prec-Recall curves
# 2) Autotune FastText and Evaluate if there are any performance gains

####################################################
#  Evaluation metrics: macro-avg F1 + accuracy     #
####################################################
# Load training and testing data
amz_train_data = pd.read_csv('./Amazon product reviews dataset/amazon_train_processed.csv')
amz_test_data = pd.read_csv('./Amazon product reviews dataset/amazon_test_processed.csv')
amz_combined_data = pd.concat([amz_train_data, amz_test_data], axis=0)


## Text Preprocessing
## A. TF-IDF
def identity_tokenizer(text):
    return text


tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)
amz_train_data_tfidf = tfidf.fit_transform(amz_train_data.text_processed)
amz_test_data_tfidf = tfidf.fit_transform(amz_test_data.text_processed)

####################################################
#  Word Embeddings: Word 2 Vec & Fast Text         #
####################################################
## B. Word2Vec (KIV - Training W2V model, unlikely to better google's pre-trained vectors)
# Load Google's pre-trained 300D word2vec model
w2v = gensim.models.KeyedVectors.load_word2vec_format(
    '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/nlpaug_model_dir/GoogleNews-vectors-negative300.bin',
    binary=True)

# w2v Training & Testing Set
w2v_mean_vectoriser = MeanEmbeddingVectorizer(w2v, 'w2v')
w2v_train_data = w2v_mean_vectoriser.fit_transform(amz_train_data)
w2v_test_data = w2v_mean_vectoriser.fit_transform(amz_test_data)

## C. FastText Embeddings - Handles out of vocab vectors using composite n_grams
# 1) Create un-tokenised text data
tqdm.pandas(desc="Progress bar")
amz_combined_data_unified = pd.DataFrame(
    amz_combined_data.progress_apply(lambda x: str(' '.join(literal_eval(x['text_processed']))), axis=1),
    columns=['text_processed'])

# 2) train and save Fast Text model
fast_text_model = FastText(window=5, min_count=5)
# Build vocab
docs = amz_combined_data_unified['text_processed'].to_list()
fast_text_model.build_vocab(sentences=docs)
# Train model
fast_text_model.train(
    sentences=docs,
    epochs=fast_text_model.epochs,
    total_examples=fast_text_model.corpus_count,
    total_words=fast_text_model.corpus_total_words
)
# fast_text_model.save('/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Word Embedding Models/fast_text.model')
fast_text_model = KeyedVectors.load(
    '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Word Embedding Models/fast_text.model')
# fast-text Training & Testing
ft_mean_vectoriser = MeanEmbeddingVectorizer(fast_text_model)
ft_train_data = ft_mean_vectoriser.fit_transform(amz_train_data)
ft_test_data = ft_mean_vectoriser.fit_transform(amz_test_data)

# 3) OR Load pre-trained fasttext vectors
fasttext_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')

# fast-text Wiki Training & Testing (use 'w2v' mode since vocab is 'fixed' using loaded Wiki vectors)
ft_wiki_mean_vectoriser = MeanEmbeddingVectorizer(fasttext_vectors, 'w2v')
ft_wiki_train_data = ft_wiki_mean_vectoriser.fit_transform(amz_train_data)
ft_wiki_test_data = ft_wiki_mean_vectoriser.fit_transform(amz_test_data)


##########################################
#        Model Development & Eval        #
##########################################
# Auxiliary Functions
def get_clf_results(y_true, y_pred):
    print(multilabel_confusion_matrix(y_true, y_pred))
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))


def tune_model(model, X, search_params, verbosity=True, n_jobs=-1):
    tuner = RandomizedSearchCV(model, search_params, verbose=verbosity, n_jobs=n_jobs)
    train_data, train_labels = X
    search = tuner.fit(train_data, train_labels)
    return search.best_estimator_, search.best_score_, search.best_params_


def save_model(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
        print('Model successfully saved at: ' + filepath)


def load_model(filepath):
    with open(filepath, 'rb') as file:
        print('Loading model from: ' + filepath)
        return pickle.load(file)


## 1. Multinomial Naive Bayes classifier (No negative values since multinomial distributions)
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

# Optimal SVC model with TFIDF
svc_clf_optimal, svc_clf_score, svc_clf_params = tune_model(SVC(),
                                                            (amz_train_data_tfidf.toarray(), amz_train_data.sentiment),
                                                            search_params=params_dict)

amz_pred_svc = svc_clf_optimal.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_pred_svc)

# Results:
# -Accuracy: 0.482423
# -Weighted F1: 0.591577
# -Macro F1: 0.317604

# Save model
save_model(svc_clf_optimal,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/svc_tfidf.pkl')

# Optimal SVC models with Word Embeddings
# a. Word2Vec
svc_w2v = SVC(kernel='linear', gamma='auto', class_weight='balanced', verbose=1)
svc_w2v.fit(w2v_train_data, amz_train_data.sentiment)
svc_w2v_pred = svc_w2v.predict(w2v_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), svc_w2v_pred)
# Results:
# -Accuracy: 0.720881
# -Weighted F1: 0.786658
# -Macro F1: 0.491202
# Improvement across the board in terms accuracy & F1 (weighted & macro) as well as across all sentiment classes

# Save W2V model
save_model(svc_w2v,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/svc_w2v.pkl')

# b. Fast Text Trained
svc_fast_txt = SVC(kernel='linear', gamma='auto', class_weight='balanced', verbose=1)
svc_fast_txt.fit(ft_train_data, amz_train_data.sentiment)
svc_fast_txt_pred = svc_fast_txt.predict(ft_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), svc_fast_txt_pred)
# Results:
# -Accuracy: 0.84371
# -Weighted F1: 0.833509
# -Macro F1: 0.348814
# Overall improvement (Esp. for positive) but significant decrease in F1 (Esp. Recall) for negative sentiments
# Observable tradeoff between majority positive class against other minority classes.

# Save W2V model
save_model(svc_w2v,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/svc_fasttxt_trained.pkl')

# c. Fast Text Loaded
svc_fast_txt_wiki = SVC(kernel='linear', gamma='auto', class_weight='balanced', verbose=1)
svc_fast_txt_wiki.fit(ft_wiki_train_data, amz_train_data.sentiment)
svc_fast_txt_loaded_pred = svc_fast_txt_wiki.predict(ft_wiki_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), svc_fast_txt_loaded_pred)
# Results:
# -Accuracy: 0.713116
# -Weighted F1: 0.781780
# -Macro F1: 0.487965
# Similar results to preloaded w2v from GoogleNews vectors. Slightly better negative sentiment accuracy whilst trading off
# with a slightly poorer positive and neutral class accuracy

# Save W2V model
save_model(svc_w2v,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/svc_fasttxt_wiki.pkl')

### SVC Conclusion ###
# Generally tradeoff between 3 classes. Word vectors increase accuracy and F1 significantly and w2v seems to be the best
# compromise in terms of F1-Macro. Limiting factor remains imbalanced data with ~95% being positive sentiment.
# Trained word embeddings generally over-fit on words and semantics of the majority class = Positive.
# Test with augmented datasets to see if results improve.

## 3. Random Forest
params_dict_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'class_weight': ['balanced']
}

# Optimal RF model with TFIDF
rf_clf_optimal, rf_clf_score, rf_clf_params = tune_model(RandomForestClassifier(),
                                                         (amz_train_data_tfidf.toarray(), amz_train_data.sentiment),
                                                         search_params=params_dict_rf)

amz_pred_rf = rf_clf_optimal.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_pred_rf)

# Results:
# -Accuracy: 0.649584
# -Weighted F1: 0.725631
# -Macro F1: 0.360333
# Strong improvement over SVC, slight dip in negative sentiment predictions.

# Optimal RF models with Word Embeddings
# a. Word2Vec
rf_w2v_optimal, rf_w2v_score, rf_w2v_params = tune_model(RandomForestClassifier(),
                                                         (w2v_train_data, amz_train_data.sentiment),
                                                         search_params=params_dict_rf)

rf_w2v_pred = rf_w2v_optimal.predict(w2v_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), rf_w2v_pred)

# Results:
# -Accuracy: 0.710716
# -Weighted F1: 0.775801
# -Macro F1: 0.460367
# Balanced results, with strong accuracy improvements in negative and neutral sentiment categories

# Save W2V model
save_model(rf_w2v_optimal,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/rf_w2v_optimal.pkl')

# b. Fast Text Trained
rf_fasttxt_optimal, rf_fasttxt_score, rf_fasttxt_params = tune_model(RandomForestClassifier(),
                                                                     (ft_train_data, amz_train_data.sentiment),
                                                                     search_params=params_dict_rf)

rf_fasttxt_pred = rf_fasttxt_optimal.predict(ft_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), rf_fasttxt_pred)

# Results:
# -Accuracy: 0.660737
# -Weighted F1: 0.735803
# -Macro F1: 0.393156
# Overall drop in performance across all sentiment categories and metrics versus word2vec model.

# Save Fast Text Trained model
save_model(rf_fasttxt_optimal,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/rf_fasttxt_trained.pkl')

# c. Fast Text Loaded
rf_fasttxt_wiki_optimal, rf_fasttxt_wiki_score, rf_fasttxt_wiki_params = tune_model(RandomForestClassifier(),
                                                                                    (ft_wiki_train_data,
                                                                                     amz_train_data.sentiment),
                                                                                    search_params=params_dict_rf)

rf_fasttxt_wiki_pred = rf_fasttxt_wiki_optimal.predict(ft_wiki_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), rf_fasttxt_wiki_pred)

# Results:
# -Accuracy: 0.740505
# -Weighted F1: 0.794781
# -Macro F1: 0.479646
# Improvements across the board (metrics and sentiment categories). Best results thus far in terms of absolute metric performance tempered by holistic performance
# across sentiment categories.

# Save Fast Text Wiki model
save_model(rf_fasttxt_wiki_optimal,
           '/Users/MacBookPro15/Documents/GitHub/Sentiment-Analysis-and-Topic-Modelling-on-Amazon-Product-Reviews/Saved Models/Sentiment Classification Models/svc_fasttxt_wiki.pkl')

### RF Conclusion ###
# Strong results when using fast text wiki data, out performs SVC in terms of Macro F1 but performs more poorly in terms of accuracy vis a vis Fast Text Trained
# SVC model. This is largely due to SVC overfitting and RF's decision boundary tending towards a more balanced performance. RF with pre-trained Fast Text word Embeddings
# thus gives us the best results so far on an severely imbalanced dataset.

##########################################
#    Improving on our Word Embeddings    #
##########################################
# 1.) Generate TFIDF weighted mean word embeddings based on our most promising neural word embeddings
# w2v
w2v_tfidf_vectoriser = TfidfEmbeddingVectorizer(w2v, 'w2v')
w2v_tfidf_train_data = w2v_tfidf_vectoriser.fit_transform(amz_train_data)
w2v_tfidf_test_data = w2v_tfidf_vectoriser.fit_transform(amz_test_data)

# fasttext loaded
fasttext_tfidf_vectoriser = TfidfEmbeddingVectorizer(fasttext_vectors)
ft_tfidf_train_data = w2v_tfidf_vectoriser.fit_transform(amz_train_data)
ft_tfidf_test_data = w2v_tfidf_vectoriser.fit_transform(amz_test_data)

### SVC ####
svc_w2v_tfidf_clf_optimal, svc_w2v_tfidf_score, svc_w2v_tfidf_params = tune_model(SVC(),
                                                                                  (w2v_tfidf_train_data,
                                                                                   amz_train_data.sentiment),
                                                                                  search_params=params_dict)

amz_svc_w2v_tfidf_pred = svc_w2v_tfidf_clf_optimal.predict(w2v_tfidf_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_svc_w2v_tfidf_pred)
# SVC Results with w2v
# - Accuracy = 0.722575
# - F2 Macro = 0.490665
# - F1 Weighted = 0.787518

svc_ft_tfidf_clf_optimal, svc_ft_tfidf_score, svc_ft_tfidf_params = tune_model(SVC(),
                                                                               (ft_tfidf_train_data,
                                                                                amz_train_data.sentiment),
                                                                               search_params=params_dict)

amz_svc_ft_tfidf_pred = svc_ft_tfidf_clf_optimal.predict(ft_tfidf_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_svc_ft_tfidf_pred)
# SVC Results with fasttext
# - Accuracy = 0.722575
# - F2 Macro = 0.490665
# - F1 Weighted = 0.787518
# Same results as w2v

### Random Forest ###
rf_w2v_tfidf_clf_optimal, rf_w2v_tfidf_score, rf_w2v_tfidf_params = tune_model(RandomForestClassifier(),
                                                                               (w2v_tfidf_train_data,
                                                                                amz_train_data.sentiment),
                                                                               search_params=params_dict_rf)

amz_rf_w2v_tfidf_pred = rf_w2v_tfidf_clf_optimal.predict(w2v_tfidf_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_rf_w2v_tfidf_pred)
# RF Results with w2v
# - Accuracy = 0.703515
# - F2 Macro = 0.456013
# - F1 Weighted = 0.770530

rf_ft_tfidf_clf_optimal, rf_ft_tfidf_score, rf_ft_tfidf_params = tune_model(RandomForestClassifier(),
                                                                            (ft_tfidf_train_data,
                                                                             amz_train_data.sentiment),
                                                                            search_params=params_dict_rf)

amz_rf_ft_tfidf_pred = rf_ft_tfidf_clf_optimal.predict(ft_tfidf_test_data)
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_rf_ft_tfidf_pred)
# RF Results with fasttext
# - Accuracy = 0.704786
# - F2 Macro = 0.456618
# - F1 Weighted = 0.771132

### Results ###
# SVC outperforms RF, however on the overall results were poorer when using TF-IDF embeddings vs simple Mean Embeddings.
# This might largely be due TF-IDF 'overfitting' or capturing too much noise from the dominant positive sentiment
# category. As such, mean embeddings are better able to denoise and predict sentiment more accurately, especially for
# the minority classes.
