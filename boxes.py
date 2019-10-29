import numpy as np
import cv2
import joblib


def rect(img, intersections):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)

    clf = joblib.load('classifier.pkl')
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(9):
        for j in range(9):
            y1 = int(intersections[j + i * 10][1] + 5)
            y2 = int(intersections[j + i * 10 + 11][1] - 5)
            x1 = int(intersections[j + i * 10][0] + 5)
            x2 = int(intersections[j + i * 10 + 11][0] - 5)

            # cv2.imwrite('/home/sahiluppal/sudoku/vals/'+str((i+1)*(j+1))+'.bmp',img[y1:y2,x1:x2])

            X = image[y1:y2,x1:x2]

            if(X.size != 0):
                X = cv2.resize(X, (36, 36))
                num = clf.predict(np.reshape(X, (1, -1)))
                if (num[0] != 0):
                    cv2.putText(img, str(num[0]), (int(intersections[j+i*10+10][0]+10),
                                                     int(intersections[j+i*10+10][1]-30)), font, 1, (225, 0, 0), 2)
                else:
                    cv2.putText(img, str(num[0]), (int(intersections[j+i*10+10][0]+10),
                                                     int(intersections[j+i*10+10][1]-15)), font, 1, (225, 0, 0), 2)

            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img
