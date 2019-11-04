import numpy as np
import cv2
import joblib


def rect(img, intersections):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)

    clf = joblib.load('classifier.pkl')
    font = cv2.FONT_HERSHEY_SIMPLEX

    img2 = img.copy()

    for i in range(9):
        for j in range(9):
            y1 = int(intersections[j + i * 10][1] + 5)
            y2 = int(intersections[j + i * 10 + 11][1] - 5)
            x1 = int(intersections[j + i * 10][0] + 5)
            x2 = int(intersections[j + i * 10 + 11][0] - 5)

            # cv2.imwrite('/home/sahiluppal/sudoku/vals/'+str((i+1)*(j+1))+'.bmp',img[y1:y2,x1:x2])
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    while True:
        cv2.imshow('image',img2)
        if cv2.waitKey(1) & 0xFF==27:
            break
    cv2.destroyAllWindows()

    prediction = list()
    if input('is it right?') == 'y':
        for i in range(9):
            for j in range(9):
                y1 = int(intersections[j + i * 10][1] + 5)
                y2 = int(intersections[j + i * 10 + 11][1] - 5)
                x1 = int(intersections[j + i * 10][0] + 5)
                x2 = int(intersections[j + i * 10 + 11][0] - 5)
                X = image[y1:y2,x1:x2]

                if(X.size != 0):
                    X = cv2.resize(X, (36, 36))
                    num = clf.predict(np.reshape(X, (1, -1)))
                    prediction.append(num[0])

    else:
        exit()

    if len(prediction) != 81:
        print('not found 81 elements')
        print('try again')
        exit()

    prediction = np.reshape(prediction,(9,9)).T
    print(prediction)

    return img,prediction
