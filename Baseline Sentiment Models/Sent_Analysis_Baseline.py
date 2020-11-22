import pandas as pd
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

####################################################
#  Evaluation metrics: macro-avg F1 + accuracy     #
####################################################

# Load training and testing data
amz_aug_train_data = pd.read_csv('../Amazon product reviews dataset/Synonym_augmented_data/amazon_syn_aug_train_processed.csv')
amz_train_data = pd.read_csv('../Amazon product reviews dataset/amazon_train_processed.csv')
amz_test_data = pd.read_csv('../Amazon product reviews dataset/amazon_test_processed.csv')

# Text Preprocessing
# A. TF-IDF
def identity_tokenizer(text):
    return text

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)
amz_aug_train_data_tfidf = tfidf.fit_transform(amz_aug_train_data.text_processed)
amz_train_data_tfidf = tfidf.fit_transform(amz_train_data.text_processed)
amz_test_data_tfidf = tfidf.fit_transform(amz_test_data.text_processed)

################ TBD ###################
# B. Word2Vec (Try training on our existing corpus)
# Load Google's pre-trained word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('../Text Preprocessing & Augmentation/nlpaug_model_dir/GoogleNews-vectors-negative300.bin', binary=True)
########################################

# Auxiliary Function for Results
def get_clf_results(y_true, y_pred):
    print(multilabel_confusion_matrix(y_true, y_pred))
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))

##########################################
#        Model Development & Eval        #
##########################################

## 1. Multinomial Naive Bayes classifier 
# Augmented Data
gnb_clf = MultinomialNB()
gnb_clf.fit(amz_aug_train_data_tfidf.toarray(), amz_aug_train_data.sentiment)
amz_aug_pred_nb = gnb_clf.predict(amz_test_data_tfidf.toarray())
get_clf_results(amz_test_data.sentiment.to_numpy(), amz_aug_pred_nb)

# Unaugmented Data
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
