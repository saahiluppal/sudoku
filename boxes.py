import numpy as np
import cv2
import tensorflow as tf

clf = tf.keras.models.load_model('keras_augumented_modelv3.h5')

def rect(img, intersections):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)
    #clf = tf.keras.models.load_model('keras_augumented_modelv3.h5')
    #font = cv2.FONT_HERSHEY_SIMPLEX


    img2 = img.copy()

    for i in range(9):
        for j in range(9):
            y1 = int(intersections[j + i * 10][1] + 5)
            y2 = int(intersections[j + i * 10 + 11][1] - 5)
            x1 = int(intersections[j + i * 10][0] + 5)
            x2 = int(intersections[j + i * 10 + 11][0] - 5)

            # cv2.imwrite('/home/sahiluppal/sudoku/vals/'+str((i+1)*(j+1))+'.bmp',img[y1:y2,x1:x2])
            #cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    prediction = list()
    for i in range(9):
        for j in range(9):
            y1 = int(intersections[j + i * 10][1] + 5)
            y2 = int(intersections[j + i * 10 + 11][1] - 5)
            x1 = int(intersections[j + i * 10][0] + 5)
            x2 = int(intersections[j + i * 10 + 11][0] - 5)
            X = image[y1:y2,x1:x2]

            if(X.size != 0):
                X = cv2.resize(X, (28, 28))
                X = X/255.0
                X = np.resize(X, (28,28,1))
                try:
                #num = clf.predict(np.resize(X,(1,28,28,1)))
                    num = clf.predict(np.reshape(X, (1, 28, 28, 1)))
                    prediction.append(np.argmax(num[0]) if np.argmax(num[0])!=10 else 0)
                except: 
                    prediction.append(0)
                #print(num[0])

    if len(prediction) != 81:
        print(prediction)
        return img, prediction, False
    else:
        prediction = np.reshape(prediction,(9,9)).T
        print(prediction)
        return img, prediction, True
