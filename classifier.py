import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ( Convolution2D, BatchNormalization, MaxPooling2D, Dropout,
                                    Flatten, Dense)

def prepare_data_fn():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    ### Processing Train -->
    indexes = np.where(y_train == 0)[0]
    total_indexes = len(indexes)
    
    X_train = np.delete(X_train, indexes, 0)
    y_train = np.delete(y_train, indexes)
    
    zeros = np.zeros(shape=(1, 28, 28))
    zeros = np.concatenate([zeros] * total_indexes, axis = 0)
    
    X_train = np.concatenate([X_train, zeros], axis=0)
    y_train = np.concatenate([y_train, [0] * total_indexes])
    
    X_train, y_train = shuffle(X_train, y_train)
    X_train = X_train.reshape(-1, 28, 28, 1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    
    ### Processing Test -->
    indexes = np.where(y_test == 0)[0]
    total_indexes = len(indexes)
    
    X_test = np.delete(X_test, indexes, 0)
    y_test = np.delete(y_test, indexes)
    
    zeros = np.zeros(shape=(1, 28, 28))
    zeros = np.concatenate([zeros] * total_indexes, axis = 0)
    
    X_test = np.concatenate([X_test, zeros], axis=0)
    y_test = np.concatenate([y_test, [0] * total_indexes])
    
    X_test, y_test = shuffle(X_test, y_test, random_state = 42)
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = prepare_data_fn()

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 12,
    zoom_range = 0.10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
)

model = Sequential([
    Convolution2D(filters = 32, kernel_size = 3, activation = 'relu',
               input_shape = (28, 28, 1)),
    BatchNormalization(),
    Convolution2D(filters = 32, kernel_size= 3, activation = 'relu'),
    BatchNormalization(),
    Convolution2D(filters = 32, kernel_size = 5, strides = 2, padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    Convolution2D(filters = 64, kernel_size = 3, activation = 'relu'),
    BatchNormalization(),
    Convolution2D(filters = 64, kernel_size = 3, activation = 'relu'),
    BatchNormalization(),
    Convolution2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    Convolution2D(filters = 128, kernel_size = 4, activation = 'relu'),
    BatchNormalization(),
    Flatten(),
    Dropout(0.4),
    Dense(10, activation = 'softmax'),
])

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                   steps_per_epoch = 937,
                   epochs = 30,
                   validation_data = (X_test, y_test),
                   callbacks = [annealer, tf.keras.callbacks.EarlyStopping(patience=5)])
                
model.save('new_model.h5')