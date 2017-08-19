'''
@ref https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
Modified.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# __configure__
batch_size = 128
num_classes = 10
epochs = 1

# __load_and_process_data__
from dataloader import DataLoader
dataloader = DataLoader()
x_train = dataloader.train_images.astype('float32')
x_test = dataloader.val_images.astype('float32')
x_train /= 255.
x_test /= 255.
y_train = dataloader.train_labels
y_test = dataloader.val_labels
x_pred = dataloader.test_images.astype('float32')
x_pred /= 255.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# __model__
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# __training__
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
# __evaluate__
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# __predict__
pred = model.predict(x_pred, batch_size=100)
print(pred)
print(pred.shape)
