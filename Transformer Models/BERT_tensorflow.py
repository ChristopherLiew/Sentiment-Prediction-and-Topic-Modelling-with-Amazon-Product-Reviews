import os
import time
from ast import literal_eval
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import optimization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report

### TBD ###
# Tidy up code and debug training accuracy = 0 issues
# Add in text preprocessing
# Try other transformers

## Configs ##
pd.options.display.max_columns = 20
tf.get_logger().setLevel('ERROR')
K = keras.backend
K.clear_session()

### Load our preprocessed text data ###
def process_and_concat_text(X):
    list_data = [literal_eval(review) for review in X]
    concat_data = [' '.join(review) for review in list_data]
    return concat_data

## Train data
raw_train_ds = pd.read_csv('./Amazon product reviews dataset/amazon_train_processed.csv')
raw_train_text, raw_train_labels = raw_train_ds['text_processed'].to_list(), raw_train_ds['sentiment'].map({-1: 0, 0: 1, 1: 2}).to_list()
raw_train_text = process_and_concat_text(raw_train_text)

## Test data
raw_test_ds = pd.read_csv('./Amazon product reviews dataset/amazon_test_processed.csv')
raw_test_text, raw_test_labels = raw_test_ds['text_processed'].to_list(), raw_test_ds['sentiment'].map({-1: 0, 0: 1, 1: 2}).to_list()
raw_test_text = process_and_concat_text(raw_test_text)

## Create Tensorflow Dataset
def create_tf_ds(X, y=None, shuffle=True):
    if y:
        # convert labels into one_hot_encoded labels
        y = keras.utils.to_categorical(y)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            return ds.shuffle(len(ds))  # Buffer Size = Size of DS for perfect shuffling
        return ds
    else:
        ds = tf.data.Dataset.from_tensor_slices(X)
        if shuffle:
            return ds.shuffle((len(ds)))
        return ds

# Create TF train ds
amz_train_ds = create_tf_ds(raw_train_text, y=raw_train_labels)
amz_test_ds = create_tf_ds(raw_test_text, y=raw_test_labels)

## Create Validation Set
VAL_SPLIT = 0.2
val_set_size = int((len(amz_train_ds) + len(amz_test_ds)) * VAL_SPLIT)  # 20% of entire DS
amz_val_ds = amz_train_ds.take(val_set_size)
amz_train_ds = amz_train_ds.skip(val_set_size)

## Cache and Create Padded Batches
# We want to pad the length of our sequences at the batch level (empty = pad to max len of that batch)
BATCH_SIZE = 32
amz_train_ds = amz_train_ds.padded_batch(BATCH_SIZE, padded_shapes=((), (3, )))
amz_val_ds = amz_val_ds.padded_batch(BATCH_SIZE, padded_shapes=((), (3, )))
amz_test_ds = amz_test_ds.padded_batch(BATCH_SIZE, padded_shapes=((), (3, )))
next(iter(amz_val_ds))

## Loading BERT model and pre-processing model
BERT_MODEL = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1'
PREPROCESSING_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2'

## Fine Tuned BERT model
def build_bert_classifier(preprocessing_model, bert_model):
    text_input = keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    bert_preprocessing_layer = hub.KerasLayer(preprocessing_model, name='bert_preprocessing')  # Truncate input to 128
    encoder_inputs = bert_preprocessing_layer(text_input)
    bert_encoder = hub.KerasLayer(bert_model, trainable=True, name='BERT_encoder')
    outputs = bert_encoder(encoder_inputs)
    pooled_output = outputs['pooled_output']  # Embedding for the entire review dataset
    net = keras.layers.Dropout(0.2)(pooled_output)
    net = keras.layers.Dense(3, activation='softmax', name='softmax_classifier')(net)
    return keras.Model(inputs=text_input, outputs=net)

## Build model (! make sure to import tensorflow text)
classifier_model = build_bert_classifier(PREPROCESSING_MODEL, BERT_MODEL)

## View model
keras.utils.plot_model(classifier_model,
                       'bert_tf_model.png',
                       show_dtype=True,
                       show_layer_names=True,
                       show_shapes=True)

## Train model
# Loss function (Cat CrossEntropy since [n_obs, n_class]; Use SparseCategoricalEntropy if it is 1D)
loss = keras.losses.CategoricalCrossentropy(from_logits=False) # Softmax applied, thus normalised.
metric = tf.metrics.Accuracy()

# Optimizer (Copy BERT pre-training process)
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(amz_train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1 * num_train_steps)  # 10% for warm up

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(loss=loss,
                         optimizer=optimizer,
                         metrics=metric)

## Fit BERT model
# Callbacks
def get_log_dir():
    root_log_dir = os.path.join(os.curdir, "Transformer Models/my_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)

RUN_LOG_DIR = get_log_dir()
tensorboard_cb = TensorBoard(RUN_LOG_DIR)
model_checkpoint_cb = ModelCheckpoint('./Transformer Models/Saved Models/my_bert_model.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20)

# Train
print(f'Training model with {BERT_MODEL}')
history = classifier_model.fit(x=amz_train_ds,
                               validation_data=amz_val_ds,
                               epochs=epochs,
                               callbacks=[early_stopping_cb, model_checkpoint_cb, tensorboard_cb])

## Evaluate Model
# Classification Results
y_pred = np.argmax(classifier_model.predict(amz_test_ds), axis=-1)
classification_rep = pd.DataFrame(classification_report(y_pred=y_pred, y_true=raw_test_labels, output_dict=True))

## Plot training and validation learning curves
# Tensorboard (BASH): tensorboard --logdir=./Transformer\ Models/my_logs --port=6006

# 1) Loss
# 2) Accuracy

## Reset
K.clear_session()
