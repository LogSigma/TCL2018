import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM

max_features=20000
maxlen=80
batch_size=32

# 1-1.

# 1-2.
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# 2. 
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
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

loss_ax.legend(loc='upper left')
acc_ax.lengend(loc='lower left')

plt.show()
