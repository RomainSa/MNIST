import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


# download MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# re-shape data to fit into CNN
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# one-hot target encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# create model with convolutional layers and compile it
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=30,
          batch_size=2048)

# read kaggle test dataset
test = np.genfromtxt('data/kaggle_mnist_test.csv',
                     delimiter=',',
                     skip_header=1)
test = test.reshape(len(test), 28, 28, 1)

# predict
predictions = np.argmax(model.predict(test), axis=1)

# submission file generation
with open('data/submission_CNN.csv', 'w+') as f:
    f.write('ImageId,Label\n')   # header
    for i, p in enumerate(predictions):
        f.write(str(i+1) + ',' + str(p) + '\n')

