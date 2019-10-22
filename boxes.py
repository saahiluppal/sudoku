import cv2

def rect(img,points):
    image = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,1)
    for i in range(9):
        for j in range(9):
            y1=int(points[j+i*10][1]+5)
            y2=int(points[j+i*10+11][1]-5)
            x1=int(points[j+i*10][0]+5)
            x2=int(points[j+i*10+11][0]-5)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    return img
