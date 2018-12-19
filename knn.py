import numpy as np
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier

# download MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = np.concatenate((X_train, X_test))
X = X.reshape(len(X), 784)
y = np.concatenate((y_train, y_test))

# read kaggle test dataset
test = np.genfromtxt('data/kaggle_mnist_test.csv',
                     delimiter=',',
                     skip_header=1)

# fitting a k-nn
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

# batch predictions
batch = 16
predictions = []
for i in range(0, len(test), batch):
    print('{}/{}'.format(i, len(test)))
    predictions += list(clf.predict(test[i:i+batch]))

# submission file generation
with open('data/submission_KNN.csv', 'w+') as f:
    f.write('ImageId,Label\n')   # header
    for i, p in enumerate(predictions):
        f.write(str(i+1) + ',' + str(p) + '\n')

