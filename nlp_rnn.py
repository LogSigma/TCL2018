import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout, Conv1D, MaxPooling1D

max_features=20000
test_max_words=80
batch_size=32

# 1-1.

# 1-2.
x_train = sequence.pad_sequences(x_train, maxlen=test_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=test_max_words)

# 2. 
model = Sequential()
model.add(Embedding(max_features, 128, input_length=test_max_words))
model.add(Droppout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 3. 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 
hist = model.fit(x_train, y_train, epochs=3, batch_size=batch_size,
                 validation_data=(x_train, y_train), verbose=2)

# 5.
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
print('Test performance: accuracy={0}, loss={1}'.format(acc, loss))

# 6. 
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
