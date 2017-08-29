
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Merge
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


BASE_DIR = ''
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = 'train_f2.csv'
TEST_DATA_FILE = 'test_f2.csv'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
'''
num_lstm = 250
num_dense = 150
rate_drop_lstm = 0.25
rate_drop_dense = 0.25
'''
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)


print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []
f6 = []
f7 = []
f8 = []
f9 =[]
f10 = []
f11=[]
f12 = []
f13 = []
f14 = []
f15 = []
f16 = []
f17 = []
f18 = []
f19 = []
f20 = []
f21 = []

with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[1]))
        texts_2.append(text_to_wordlist(values[2]))
        labels.append(int(values[3]))
	f1.append(float(values[4]))
	f2.append(float(values[5]))
	f3.append(float(values[6]))
	f4.append(float(values[7]))
	f5.append(float(values[8]))
        f6.append(float(values[9]))
        f7.append(float(values[10]))
        f8.append(float(values[11]))
	f9.append(float(values[12]))
        f10.append(float(values[13]))
        f11.append(float(values[14]))
        f12.append(float(values[15]))
	f13.append(float(values[16]))
        f14.append(float(values[17]))
        f15.append(float(values[18]))
        f16.append(float(values[19]))
        f17.append(float(values[20]))
        f18.append(float(values[21]))
        f19.append(float(values[22]))
        f20.append(float(values[23]))
        f21.append(float(values[24]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
tf1 = []
tf2 = []
tf3 = []
tf4 = []
tf5 = []
tf6 = []
tf7 = []
tf8 = []
tf9 =[]
tf10 = []
tf11=[]
tf12 = []
tf13 = []
tf14 = []
tf15 = []
tf16 = []
tf17 = []
tf18 = []
tf19 = []
tf20 = []
tf21 = []

with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
	tf1.append(float(values[3]))
        tf2.append(float(values[4]))
        tf3.append(float(values[5]))
        tf4.append(float(values[6]))
        tf5.append(float(values[7]))
        tf6.append(float(values[8]))
        tf7.append(float(values[9]))
        tf8.append(float(values[10]))
        tf9.append(float(values[11]))
        tf10.append(float(values[12]))
        tf11.append(float(values[13]))
        tf12.append(float(values[14]))
        tf13.append(float(values[15]))
        tf14.append(float(values[16]))
        tf15.append(float(values[17]))
        tf16.append(float(values[18]))
        tf17.append(float(values[19]))
        tf18.append(float(values[20]))
        tf19.append(float(values[21]))
        tf20.append(float(values[22]))
        tf21.append(float(values[23]))
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

f_tr = []
for i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21 in zip(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21):
	f_tr.append([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21])
f_te = []
for i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21 in zip(tf1,tf2,tf3,tf4,tf5,tf6,tf7,tf8,tf9,tf10,tf11,tf12,tf13,tf14,tf15,tf16,tf17,tf18,tf19,tf20,tf21):
        f_te.append([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21])
f_tr = np.array(f_tr)
f_te = np.array(f_te)


print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
f_tr1 = np.vstack((f_tr[idx_train], f_tr[idx_train]))
#f_tr1 = f_tr[:idx_train]

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))
f_v = np.vstack((f_tr[idx_val], f_tr[idx_val]))
#f_tr1 = f_tr[:idx_train]

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344


'''
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)
'''
model1 = Sequential()
model1.add(Embedding(nb_words,
                     EMBEDDING_DIM,
                     weights = [embedding_matrix],
                     input_length = MAX_SEQUENCE_LENGTH,
                     trainable = False))
model1.add(LSTM(num_lstm,dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
model2 = Sequential()
model2.add(Embedding(nb_words,
                     EMBEDDING_DIM,
                     weights = [embedding_matrix],
                     input_length = MAX_SEQUENCE_LENGTH,
                     trainable = False))
model2.add(LSTM(num_lstm,dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

modelc = Sequential()
modelc.add(Dense(num_dense + 10, input_shape=(21,)))
modelc.add(Activation('relu'))
modelc.add(Dropout(rate_drop_dense))
modelc.add(BatchNormalization())
model = Sequential()
model.add(Merge([model1, model2, modelc], mode='concat'))
#modela.add(Dense(units*2))
model.add(Dropout(rate_drop_dense))
model.add(BatchNormalization())
model.add(Dense(num_dense))
model.add(Activation('relu'))
model.add(Dropout(rate_drop_dense))
model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation('sigmoid'))

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None



model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=16)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train, f_tr1], labels_train, \
        validation_data=([data_1_val, data_2_val, f_v], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])


print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2, f_te], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, f_te], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('lstm_withFE_26thApr_ngrams1.csv', index=False)
