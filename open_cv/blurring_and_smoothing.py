import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    black = [[180, 255, 30], [0, 0, 0]]
    white = [[180, 18, 255], [0, 0, 231]]
    red = [[180, 255, 255], [159, 50, 70]]
    green = [[89, 255, 255], [36, 50, 70]]
    blue = [[128, 255, 255], [90, 50, 70]]
    yellow = [[35, 255, 255], [25, 50, 70]]
    purple = [[158, 255, 255], [129, 50, 70]]
    orange = [[24, 255, 255], [10, 50, 70]]
    gray = [[180, 18, 230], [0, 0, 40]]

    upper_color = np.array(white[0])
    lower_color = np.array(white[1])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask= mask)

    kernel = np.ones((15, 15), np.float32)/225
    
    smoothed = cv2.filter2D(result, -1, kernel)
    blur = cv2.GaussianBlur(result, (15, 15), 0)
    median = cv2.medianBlur(result, 15)
    bilateral = cv2.bilateralFilter(result, 15, 75, 75)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.imshow('smoothed', smoothed)
    cv2.imshow('blur', blur)
    cv2.imshow('median', median)
    cv2.imshow('bilateral', bilateral)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()