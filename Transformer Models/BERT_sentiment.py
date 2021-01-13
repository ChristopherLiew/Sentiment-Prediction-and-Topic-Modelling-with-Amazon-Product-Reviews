import os
import time
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses, layers, utils, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from transformers import TFBertModel, BertTokenizer, BertConfig
import pandas as pd
import numpy as np

### TBD ###
# Review BERT model output
# Fine tune BERT + Try other transformer architectures
# Add in text preprocessing

### Load our datasets ###
## Train data
raw_train_ds = pd.read_csv('./Amazon product reviews dataset/Labelled_Amazon_Reviews/train/amazon_train.csv')
raw_train_text, raw_train_labels = raw_train_ds['reviews.text'].to_list(), raw_train_ds['sentiment'].map(
    {-1: 0, 0: 1, 1: 2}).to_list()

## Test data
raw_test_ds = pd.read_csv('Amazon product reviews dataset/Labelled_Amazon_Reviews/test/amazon_test.csv')
raw_test_text, raw_test_labels = raw_test_ds['reviews.text'].to_list(), raw_test_ds['sentiment'].map(
    {-1: 0, 0: 1, 1: 2}).to_list()

## Configs
TRANSFORMER_MODEL = 'bert-base-uncased'

## Tokenize our Train and Test sets
bert_config = BertConfig.from_pretrained(TRANSFORMER_MODEL)
bert_tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL, do_lower_case=True)

# Train Set
encoded_train_text = bert_tokenizer(raw_train_text,
                                    padding=True,
                                    add_special_tokens=True,
                                    truncation=True)

# View decoded train text as a sanity check
decoded_train_text = [bert_tokenizer.decode(ids) for ids in encoded_train_text['input_ids']]

# Test Set
encoded_test_text = bert_tokenizer(raw_test_text,
                                   padding=True,
                                   add_special_tokens=True,
                                   truncation=True)

# View decoded test text as a sanity check
decoded_test_text = [bert_tokenizer.decode(ids) for ids in encoded_test_text['input_ids']]

### Load pretrained BERT model and tokenizer ###
## Set up our Baseline BERT model on unprocessed datasets
def create_BERT_model(bert_model_name, bert_model_config, num_classes, lr=0.001):
    # Input layer
    input_ids = layers.Input(shape=(bert_model_config.max_position_embeddings, ), dtype=tf.int32, name='input_ids')
    token_type_ids = layers.Input(shape=(bert_model_config.max_position_embeddings, ), dtype=tf.int32,
                                  name='token_type_ids')
    attention_mask = layers.Input(shape=(bert_model_config.max_position_embeddings, ), dtype=tf.int32,
                                  name='attention_mask')
    # Instantiate BERT layer
    bert_model = TFBertModel.from_pretrained(bert_model_name, config=bert_model_config)
    # Feed input into BERT layer (hidden_size = 768)
    bert_embeddings = bert_model(input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask)[0][:, 0, :]  # Keep only CLS tokens (Review this again)
    # Attach 1-layer Feed Forward NN (hidden_size = num_classes)
    dropout = layers.Dropout(bert_config.hidden_dropout_prob, name='ff_dropout_layer')(bert_embeddings)
    hidden_layer = layers.Dense(num_classes, name='ff_hidden_layer')(dropout)
    sentiment_probs = layers.Activation(activations.softmax, name='softmax_output')(hidden_layer)
    model = Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[sentiment_probs],
        name='BERT_sentiment_classifier'
    )
    # Loss & Optimizer
    loss = losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

amz_bert_model = create_BERT_model('bert-base-uncased', bert_config, num_classes=3, lr=0.01)
amz_bert_model.summary()
utils.plot_model(amz_bert_model, 'bert_base_model.png',
                 show_layer_names=True,
                 show_dtype=True,
                 show_shapes=True)

## Callbacks
# Tensorboard
def get_log_dir():
    root_log_dir = os.path.join(os.curdir, "Transformer Models/my_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)

tensorboard_cb = TensorBoard(get_log_dir())
# Early Stopping
early_stopping_cb = EarlyStopping(patience=20)

## Train Model (When fitting multi inputs pass in a dict with keys mapping to each input layer)
history = amz_bert_model.fit(x={'input_ids': np.asarray(encoded_train_text['input_ids']),
                                'token_type_ids': np.asarray(encoded_train_text['token_type_ids']),
                                'attention_mask': np.asarray(encoded_train_text['attention_mask'])
                                },
                             y={'softmax_output': np.asarray(raw_train_labels)},
                             epochs=2,
                             verbose=2,
                             callbacks=[tensorboard_cb, early_stopping_cb])

## Evaluate Model
amz_bert_model.evaluate(x={'input_ids': np.asarray(encoded_test_text['input_ids']),
                           'token_type_ids': np.asarray(encoded_test_text['token_type_ids']),
                           'attention_mask': np.asarray(encoded_test_text['attention_mask'])
                           },
                        y={'softmax_output': np.asarray(raw_test_labels)},
                        return_dict=True)

### TENSORFLOW DF ###
## Create Tensorflow Dataset
def create_tf_ds(encodings, y=None, shuffle=True):
    if y:
        ds = tf.data.Dataset.from_tensor_slices((dict(encodings), y))
        if shuffle:
            return ds.shuffle(len(ds))  # Buffer Size = Size of DS for perfect shuffling
        return ds
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(encodings))
        if shuffle:
            return ds.shuffle((len(ds)))
        return ds

# Create TF train ds
amz_train_ds = create_tf_ds(encoded_train_text, y=raw_train_labels)
amz_test_ds = create_tf_ds(encoded_test_text, y=raw_test_labels)

## Create Validation Set
val_split = 0.2
val_set_size = int((len(amz_train_ds) + len(amz_test_ds)) * val_split)  # 20% of entire DS

# Validation set
amz_val_ds = amz_train_ds.take(val_set_size)
amz_train_ds = amz_train_ds.skip(val_set_size)

## Cache and Create Batches
BATCH_SIZE = 32
amz_train_ds = amz_train_ds.batch(BATCH_SIZE)
amz_val_ds = amz_val_ds.batch(BATCH_SIZE)
amz_test_ds = amz_test_ds.batch(BATCH_SIZE)
