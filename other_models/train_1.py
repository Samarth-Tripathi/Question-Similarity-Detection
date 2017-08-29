
from __future__ import print_function
import numpy as np
import pandas as pd
import datetime, time, json
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
#from sklearn.model_selection import train_test_split

import csv, json
from zipfile import ZipFile
from os.path import expanduser, exists

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
import codecs



Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.01
TEST_SPLIT = 0.01
RNG_SEED = 13371447
NB_EPOCHS = 20

q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
with open(NB_WORDS_DATA_FILE, 'r') as f:
    nb_words = json.load(f)['nb_words']

X = np.stack((q1_data, q2_data), axis=1)
y = labels
print (X.shape)
print (y.shape)

X_train = X
y_train = y
X_test = X[:5000]
y_test = y[:5000]
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

Q1 = Sequential()
Q1.add(Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=MAX_SEQUENCE_LENGTH,
                 trainable=False))
Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
Q2 = Sequential()
Q2.add(Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=MAX_SEQUENCE_LENGTH,
                 trainable=False))
Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
model = Sequential()
model.add(Merge([Q1, Q2], mode='concat'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
history = model.fit([Q1_train, Q2_train],
                    y_train,
                    nb_epoch=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=2,
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
print("Saving model...")
model.save("Model_1")

question1 = []
question2 = []
test_id = []
counter = 0
with codecs.open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        question1.append(row['question1'])
        question2.append(row['question2'])
        test_id.append(row['test_id'])

print('Question pairs: %d' % len(question1))

KERAS_DATASETS_DIR = '/datasets/'
#QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
#QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
#GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip [ nlp. stanford. edu/data/glove. 840B. 300d. zip ] '
#GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
#Q1_TRAINING_DATA_FILE = 'q1_train.npy'
#Q2_TRAINING_DATA_FILE = 'q2_train.npy'
#LABEL_TRAINING_DATA_FILE = 'label_train.npy'
#WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
#NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300

questions = question1 + question2
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index

print("Words in index: %d" % len(word_index))

Q1_TEST_DATA_FILE = 'q1_test.npy'
Q2_TEST_DATA_FILE = 'q2_test.npy'
LABEL_TEST_DATA_FILE = 'label_test.npy'
WORD_EMBEDDING_MATRIX_FILE_TEST = 'word_embedding_matrix_test.npy'
NB_WORDS_DATA_FILE_TEST = 'nb_words_test.json'

print("Processing", GLOVE_FILE)
KERAS_DATASETS_DIR = '/datasets/'
embeddings_index = {}
with open('datasets/glove.840B.300d.txt') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings: %d' % len(embeddings_index))

nb_words = 105000
#nb_words = min(MAX_NB_WORDS, len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_id = np.array(test_id, dtype=int)
print('Shape of question1 data tensor:', q1_data.shape)
print('Shape of question2 data tensor:', q2_data.shape)
print('Shape of label tensor:', test_id.shape)

np.save(open(Q1_TEST_DATA_FILE, 'wb'), q1_data)
np.save(open(Q2_TEST_DATA_FILE, 'wb'), q2_data)
np.save(open(LABEL_TEST_DATA_FILE, 'wb'), test_id)
np.save(open(WORD_EMBEDDING_MATRIX_FILE_TEST, 'wb'), word_embedding_matrix)
with open(NB_WORDS_DATA_FILE_TEST, 'w') as f:
    json.dump({'nb_words_test': nb_words}, f)

q1_data = np.load(open(Q1_TEST_DATA_FILE, 'rb'))
q2_data = np.load(open(Q2_TEST_DATA_FILE, 'rb'))
ids = np.load(open(LABEL_TEST_DATA_FILE, 'rb'))
word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE_TEST, 'rb'))
with open(NB_WORDS_DATA_FILE_TEST, 'r') as f:
    nb_words = json.load(f)['nb_words_test']

X = np.stack((q1_data, q2_data), axis=1)
y = ids
X_test = X
y_test = y
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

prediction = model.predict([Q1_test,Q2_test], verbose = 1)

with open('predictions_1.csv', 'wb') as csvfile:
    out_writer = csv.writer(csvfile, delimiter=',')
    out_writer.writerow(['test_id','is_duplicate'])
    for tid,tout in zip(y_test,prediction):
        out_writer.writerow([tid, tout[0]])
