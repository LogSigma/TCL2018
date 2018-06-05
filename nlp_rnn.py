import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from konlpy.tag import Twitter
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout, Conv1D, MaxPooling1D

twitter = Twitter()
max_features=20000
text_max_words=80
batch_size=32

# 0. 함수 선언
def tokenize(doc):
  return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]

def build_vocab(tokens):
  vocab = dict()
  vocab['#UNKOWN'] = 0
  vocab['#PAD'] = 1
  for t in tokens:
    if t not in vocab:
      vocab[t] = len(vocab)
  return vocab

def get_token_id(token, vocab):
  if token in vocab:
    return vocab[token]
  else:
    0

def build_input(data, vocab):
  result = []
  for d in data:
    seq = [get_token_id(t, vocab) for t in d[0]]
    while len(seq) > 0:
      seq_seg = seq[:text_max_words]
      seq = seq[text_max_words:]
      
      padding = [1] *(text_max_words - len(seq_seg))
      seq_seg = seq_seg + padding
      result.append(seq_seg)
  return result

# 1. 데이터셋 생성하기

# 
train = pd.read_csv('data/ratings_train.txt', sep='\t', encoding='CP949')
test = pd.read_csv('data/ratings_test.txt', sep='\t', encoding='CP949')

# 
train_data = [tokenize(row[2]) for row in train.itertuples()]
test_data = [tokenize(row[2]) for row in test.itertuples()]
#train_data = list(train['document'].apply(lambda x: ['/'.join(t) for t in twitter.pos(x, norm=True, stem=True)]))

#  
tokens = [t for d in train_data for t in d]
vocab = build_vocab(tokens)

train_data = build_input(train_data, vocab)
test_data = build_input(test_data, vocab)

# 
x_train = np.array(train_data)
y_train = np.array(train['label'])
x_test = np.array(test_data)
y_test = np.array(test['label'])

# 2. 모델 구성하기 
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=3, batch_size=batch_size,
                 validation_data=(x_train, y_train), verbose=2)

# 5. 모델 평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
print('Test performance: accuracy={0}, loss={1}'.format(acc, loss))

# 6. 학습과정 
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='lower left')
acc_ax.legend(loc='upper left')

plt.show()
