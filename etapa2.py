import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

train = pd.read_csv('mitbih_train.csv', header=None)
test = pd.read_csv('mitbih_test.csv', header=None)

ytrain = to_categorical(train[train.columns[-1]], num_classes=5)
ytest = to_categorical(test[test.columns[-1]], num_classes=5)

# MLP
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(187,)))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(train[train.columns[:-1]], ytrain, epochs=10,
                    batch_size=128, validation_data=(test[test.columns[:-1]], ytest))
predictions = model.predict(test[test.columns[:-1]])
ypred = pd.DataFrame(tf.argmax(predictions, axis=1))
r = classification_report(test[test.columns[-1]], ypred)
a = accuracy_score(test[test.columns[-1]], ypred)
cm = confusion_matrix(test[test.columns[-1]], ypred)
print(a)
print(cm)
print(r)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Testare')
plt.xlabel('Numar Epoci')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Conv1D
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3,
          activation='relu', input_shape=(187, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling1D())
model.add(Dense(units=5, activation='softmax'))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(train[train.columns[:-1]], ytrain, epochs=10,
                    batch_size=128, validation_data=(test[test.columns[:-1]], ytest))
predictions = model.predict(test[test.columns[:-1]])
ypred = pd.DataFrame(tf.argmax(predictions, axis=1))
r = classification_report(test[test.columns[-1]], ypred)
a = accuracy_score(test[test.columns[-1]], ypred)
cm = confusion_matrix(test[test.columns[-1]], ypred)
print(a)
print(cm)
print(r)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Testare')
plt.xlabel('Numar Epoci')
plt.ylabel('Loss')
plt.legend()
plt.show()

# LSTM
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(187, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(train[train.columns[:-1]], ytrain, epochs=10,
                    batch_size=128, validation_data=(test[test.columns[:-1]], ytest))
predictions = model.predict(test[test.columns[:-1]])
ypred = pd.DataFrame(tf.argmax(predictions, axis=1))
r = classification_report(test[test.columns[-1]], ypred)
a = accuracy_score(test[test.columns[-1]], ypred)
cm = confusion_matrix(test[test.columns[-1]], ypred)
print(a)
print(cm)
print(r)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Testare')
plt.xlabel('Numar Epoci')
plt.ylabel('Loss')
plt.legend()
plt.show()


ptbdb_normal = pd.read_csv('ptbdb_normal.csv', header=None)
ptbdb_abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)
ptbdb = pd.concat([ptbdb_normal, ptbdb_abnormal])

train, test = train_test_split(ptbdb, test_size=0.2, random_state=42)

ytrain = to_categorical(train[train.columns[-1]], num_classes=5)
ytest = to_categorical(test[test.columns[-1]], num_classes=5)

# MLP
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(187,)))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(train[train.columns[:-1]], ytrain, epochs=10,
                    batch_size=32, validation_data=(test[test.columns[:-1]], ytest))
predictions = model.predict(test[test.columns[:-1]])
ypred = pd.DataFrame(tf.argmax(predictions, axis=1))
r = classification_report(test[test.columns[-1]], ypred)
a = accuracy_score(test[test.columns[-1]], ypred)
cm = confusion_matrix(test[test.columns[-1]], ypred)
print(a)
print(cm)
print(r)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Testare')
plt.xlabel('Numar Epoci')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Conv1D
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3,
          activation='relu', input_shape=(187, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling1D())
model.add(Dense(units=5, activation='softmax'))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(train[train.columns[:-1]], ytrain, epochs=10,
                    batch_size=32, validation_data=(test[test.columns[:-1]], ytest))
predictions = model.predict(test[test.columns[:-1]])
ypred = pd.DataFrame(tf.argmax(predictions, axis=1))
r = classification_report(test[test.columns[-1]], ypred)
a = accuracy_score(test[test.columns[-1]], ypred)
cm = confusion_matrix(test[test.columns[-1]], ypred)
print(a)
print(cm)
print(r)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Testare')
plt.xlabel('Numar Epoci')
plt.ylabel('Loss')
plt.legend()
plt.show()

# LSTM
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(187, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(train[train.columns[:-1]], ytrain, epochs=10,
                    batch_size=32, validation_data=(test[test.columns[:-1]], ytest))
predictions = model.predict(test[test.columns[:-1]])
ypred = pd.DataFrame(tf.argmax(predictions, axis=1))
r = classification_report(test[test.columns[-1]], ypred)
a = accuracy_score(test[test.columns[-1]], ypred)
cm = confusion_matrix(test[test.columns[-1]], ypred)
print(a)
print(cm)
print(r)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Testare')
plt.xlabel('Numar Epoci')
plt.ylabel('Loss')
plt.legend()
plt.show()
