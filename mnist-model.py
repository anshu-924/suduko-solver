import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Input

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
print(X_train.shape)
import matplotlib.pyplot as plt
plt.imshow(X_train[0])
print("end")

X_train = X_train/255
X_test = X_test/255
print(X_train.shape)
model = Sequential()


model.add(Input(shape=(28, 28)))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])