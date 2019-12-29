import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

def load_and_preprocess():
    (X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    zeros = np.zeros((1,28,28))
    for i in range(12):
        zeros = np.vstack((zeros, zeros))
    X_train = np.vstack((X_train, zeros))
    y_train = np.concatenate((y_train,[10]*4096))

    zeros = np.zeros((1,28,28))
    for i in range(9):
        zeros = np.vstack((zeros, zeros))
    X_test = np.vstack((X_test, zeros))
    y_test = np.concatenate((y_test,[10]*512))

    X_train, y_train = shuffle(X_train,y_train)
    X_test, y_test = shuffle(X_test, y_test)

    X_train = X_train.reshape(-1,28,28,1)
    print('X_train.shape=',X_train.shape)
    print('y_train.shape=',y_train.shape)

    X_test = X_test.reshape(-1,28,28,1)
    print('X_test.shape=',X_test.shape)
    print('y_test.shape=',y_test.shape)

    return (X_train,y_train), (X_test, y_test)

print('Dataset Info::')
(X_train, y_train), (X_test, y_test) = load_and_preprocess()

model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu
                                 , input_shape=(28,28,1)),
    tf.keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),

    tf.keras.layers.Convolution2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Convolution2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),

    tf.keras.layers.Convolution2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Convolution2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),

    tf.keras.layers.Convolution2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Convolution2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(11, activation=tf.nn.softmax),
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,
             metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping()
model.fit(X_train,y_train, epochs=10, batch_size=128, callbacks=[es],validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)

model.save('classifier_new.h5')
