import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import nlpaug.augmenter.word as naw

# Set up MODELS dir
from nlpaug.util.file.download import DownloadUtil
dest_dir = '../Text Preprocessing & Augmentation/nlpaug_model_dir'
# DownloadUtil.download_word2vec(dest_dir=dest_dir)  # Download word2vec model
# DownloadUtil.download_glove(model_name='glove.6B', dest_dir=dest_dir)  # Download GloVe model
# DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir=dest_dir)  # Download fasttext model

amz = pd.read_csv('../Amazon product reviews dataset/Amazon_product_review_with_sent.csv')

# Augmentation should only be applied to our training set and be left out of our validation and test sets
# Split our datasets into training, validation and testing sets
amz_X = amz['reviews.text']
amz_Y = amz['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1}) # Encode sentiments
amz_X_train, amz_X_test, amz_Y_train, amz_Y_test = train_test_split(amz_X, amz_Y, test_size=0.25, random_state=42, shuffle=True)

# Train Set
amz_train_set = pd.merge(left=amz_X_train, right=amz_Y_train, left_index=True, right_index=True)
# Test Set
amz_test_set = pd.merge(left=amz_X_test, right=amz_Y_test, left_index=True, right_index=True)
# Save for future use
amz_train_set.to_csv('../Amazon product reviews dataset/amazon_train.csv', index=False)
amz_test_set.to_csv('../Amazon product reviews dataset/amazon_test.csv', index=False)

# Filter out our minority clases (i.e. Neutral & Negative sentiments)
amz_neu = amz_train_set.loc[amz_train_set['sentiment'] == 0, ['reviews.text', 'sentiment']]
amz_neg = amz_train_set.loc[amz_train_set['sentiment'] == -1, ['reviews.text', 'sentiment']]
# Majority class
amz_pos = amz_train_set.loc[amz_train_set['sentiment'] == 1, ['reviews.text', 'sentiment']]

# NLP Augmentation
model_dir = dest_dir

# Determine number of augmentations we need per minority class to balance the dataset
amz_train_set.sentiment.value_counts()

# No. of augments to perform per sample
def num_augments_per_sample(curr, target):
    return int((target - curr)/ curr)

num_augments_per_sample(1185, 19161) # Negative: ~15
num_augments_per_sample(903, 19161)  # Neutral: ~20

# 1. SynonymAug
def data_augment_synonym(data, num_augments):
    ''' 
    Augments text data by substituting words in our text with a synonym found via NLTK's WordNet. 

    params:
    data - Pandas dataframe comprising text data and labels

    output:
    augmented_dataset = Shuffled and augmented data
    '''
    working_data = data.reset_index(drop=True).copy()
    aug = naw.SynonymAug(aug_src='wordnet')
    for row in working_data.itertuples():
        review_text = row[1]
        review_sent = row[2]
        for i in range(num_augments):
            aug_text = aug.augment(review_text)
            working_data = working_data.append(pd.Series({'reviews.text': aug_text, 'sentiment': review_sent}), ignore_index=True)
    return shuffle(working_data)

amz_neu_syn_aug = data_augment_synonym(amz_neu, 20)
amz_neg_syn_aug = data_augment_synonym(amz_neg, 15)

# Combine and write augmented data
amz_syn_aug_data = amz_pos.append(amz_neu_syn_aug, ignore_index=True)
amz_syn_aug_data = amz_syn_aug_data.append(amz_neg_syn_aug, ignore_index=True)
amz_syn_aug_data = shuffle(amz_syn_aug_data)
amz_syn_aug_data.reset_index(drop=True, inplace=True) # 57084 rows
amz_syn_aug_data.to_csv('../Amazon product reviews dataset/Synonym_augmented_data/amazon_synaug_train.csv', index=False)


###########################
#       To Be Done        #
###########################

# 2. BacktranslationAug (From EN -> DE -> EN, Issue with translation file DEBUG)
text = 'The quick brown fox jumped over the lazy dog'
back_translation_aug = naw.BackTranslationAug(
    from_model_name='transformer.wmt19.en-de', to_model_name='transformer.wmt19.de-en')
back_translation_aug.augment(text)

# 3. ContextualWordEmbsAug (Takes super long so KIV)

def data_augment_CWEB_aug(data, num_augments, action='substitute'):
    working_data = data.reset_index(drop=True).copy()
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action=action)
    for row in working_data.itertuples():
        review_text = row[1]
        review_sent = row[2]
        for i in range(num_augments):
            aug_text = aug.augment(review_text)
            working_data = working_data.append(pd.Series(
                {'reviews.text': aug_text, 'sentiment': review_sent}), ignore_index=True)
    return shuffle(working_data)


amz_neu_CWEB_aug = data_augment_CWEB_aug(amz_neu, 20)
amz_neg_CWEB_aug = data_augment_CWEB_aug(amz_neg, 15)
