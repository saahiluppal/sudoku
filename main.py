import cv2
import numpy as np
from filter import filter_one
from filter import filter_two
from angle import segment_by_angle_kmeans
from intersection import segmented_intersections
from boxes import rect
from contours import max_area
import joblib
import sudoku

#img = cv2.imread('sudoku.jpeg')
cap = cv2.VideoCapture(0)

filter = True

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    if lines is None:
        continue

    if filter:
        line_flags = filter_one(lines)

    #print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        filtered_lines = filter_two(lines,line_flags)

        #print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    segmented = segment_by_angle_kmeans(filtered_lines)
    intersections = segmented_intersections(segmented) # points


    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    try:
        if len(intersections) == 100 and len(filtered_lines) == 20:
            print('Got')
            break
    except:
        pass

intersections = [val[0] for val in intersections]
intersections = sorted(intersections,key=lambda val: val[0])

image,prediction = rect(img,intersections)

while True:
    cv2.imshow('img',image)
    if cv2.waitKey(2) & 0xFF == 27:
        break

cv2.destroyAllWindows()

if sudoku.SolveSudoku(prediction):
    print(sudoku.SolveSudoku(prediction))
else:
    print('bad luck')
