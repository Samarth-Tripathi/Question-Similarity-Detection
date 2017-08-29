
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

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation

Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.025
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25

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

from keras.layers import Activation, Flatten, Convolution1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU
units = 128 # Number of nodes in the Dense layers
dropout = 0.25 # Percentage of nodes to drop
nb_filter = 32 # Number of filters to use in Convolution1D
filter_length = 3 # Length of filter for Convolution1D
# Initialize weights and biases for the Dense layers
#weights = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2)
#bias = bias_initializer='zeros'
embedding_dim = EMBEDDING_DIM
max_question_len = MAX_SEQUENCE_LENGTH

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

model1 = Sequential()

model1.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))

model1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(embedding_dim,)))

model2 = Sequential()

model2.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))

model2.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(embedding_dim,)))

model3 = Sequential()

model3.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(embedding_dim))
model3.add(Dropout(0.2))
#model3.add(BatchNormalization())

model4 = Sequential()

model4.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(embedding_dim))
model4.add(Dropout(0.2))
#model4.add(BatchNormalization())
model5 = Sequential()
model5.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))
#model5.add(Embedding(nb_words + 1, embedding_dim, input_length=max_question_lenth, dropout=0.2))
model5.add(LSTM(embedding_dim, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))
#model6.add(Embedding(nb_words + 1, embedding_dim, input_length=max_question_lenth, dropout=0.2))
model6.add(LSTM(embedding_dim, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))



merged_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
history = merged_model.fit([Q1_train, Q2_train, Q1_train, Q2_train, Q1_train, Q2_train ],
                    y_train,
                    nb_epoch=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=2,
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

def text_to_wordlist(text, remove_stop_words=False, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    
    # Convert words to lower case and split them
    #text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)





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

all_question1 = []
for question in question1:
        all_question1.append(text_to_wordlist(question))
question1 = all_question1

all_question2 = []
for question in question2:
        all_question2.append(text_to_wordlist(question))
question2 = all_question2


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
MAX_SEQUENCE_LENGTH = 50
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

prediction = merged_model.predict([Q1_test,Q2_test,Q1_test,Q2_test,Q1_test,Q2_test], verbose = 1)

with open('predictions_5.csv', 'wb') as csvfile:
    out_writer = csv.writer(csvfile, delimiter=',')
    out_writer.writerow(['test_id','is_duplicate'])
    for tid,tout in zip(y_test,prediction):
        out_writer.writerow([tid, tout[0]])
