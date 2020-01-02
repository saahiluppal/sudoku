import tensorflow as tf

model_list = ['classifier1.h5','classifier2.h5','classifier3.h5','classifier4.h5']

(_,_),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test/255.0
X_test = X_test.reshape(-1,28,28,1)

for m in model_list:
    model = tf.keras.models.load_model(m)
    history = model.evaluate(X_test, y_test)
    print(m, history[0], history[1])