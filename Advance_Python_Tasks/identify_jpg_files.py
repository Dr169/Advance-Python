import os
import cv2
import numpy as np

if not os.path.exists("/home/dr169/Desktop/task/gray-scale/"):
    os.makedirs("/home/dr169/Desktop/task/gray-scale/")
      
if not os.path.exists("/home/dr169/Desktop/task/white-round/"):
    os.makedirs("/home/dr169/Desktop/task/white-round/")
    
if not os.path.exists("/home/dr169/Desktop/task/line-detect/"):
    os.makedirs("/home/dr169/Desktop/task/line-detect/")
    
for image in os.listdir("/home/dr169/Desktop/task/"):
    if image.endswith(".jpg"):
        img = cv2.imread("/home/dr169/Desktop/task/"+image)
        height, width = img.shape[:2]
        if height > 480 and width > 640:            
            gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_scale_img = np.delete(gray_scale_img, range(1, height, 2), axis=0)
            gray_scale_img = np.delete(gray_scale_img, range(1, width, 2), axis=1)
            cv2.imwrite("/home/dr169/Desktop/task/gray-scale/"+image, gray_scale_img)  
            
for image in os.listdir("/home/dr169/Desktop/task/gray-scale/"):
    white_round_img = cv2.imread("/home/dr169/Desktop/task/gray-scale/" + image)
    height, width = white_round_img.shape[:2]
    img_array = np.ones((height+2, width+2, 3), dtype=int) * 255
    img_array[1:-1,1:-1] = white_round_img
    cv2.imwrite("/home/dr169/Desktop/task/white-round/"+image, img_array)
    
y_line_matrix = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
x_line_matrix = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

list_of_white_round_img = []
list_of_gray_img = []

for white_round_img in os.listdir("/home/dr169/Desktop/task/white-round/")[:1]:
    img = cv2.imread("/home/dr169/Desktop/task/white-round/" + white_round_img)
    img_2D = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    list_of_white_round_img.append(img_2D)
    
for line_img in os.listdir("/home/dr169/Desktop/task/gray-scale/"):
    img = cv2.imread("/home/dr169/Desktop/task/gray-scale/" + line_img)
    img_2D = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    list_of_gray_img.append(img_2D)
    
line_array_y = np.array([],dtype=int)
line_array_x = np.array([],dtype=int)

for y in range(list_of_white_round_img[0].shape[0]-2):
    for x in range(list_of_white_round_img[0].shape[1]-2):
        if np.sum(list_of_white_round_img[0][y:y+3,x:x+3] * y_line_matrix) < 0:
            line_array_y = np.append(line_array_y, 0)
            
        elif np.sum(list_of_white_round_img[0][y:y+3,x:x+3] * y_line_matrix) > 255:
            line_array_y = np.append(line_array_y, 255)
            
        else:
            line_array_y = np.append(line_array_y,np.sum(list_of_white_round_img[0][y:y+3,x:x+3] * y_line_matrix))
            
for y in range(list_of_white_round_img[0].shape[0]-2):
    for x in range(list_of_white_round_img[0].shape[1]-2):
        if np.sum(list_of_white_round_img[0][y:y+3,x:x+3] * x_line_matrix) < 0:
            line_array_x = np.append(line_array_x, 0)
            
        elif np.sum(list_of_white_round_img[0][y:y+3,x:x+3] * x_line_matrix) > 255:
            line_array_x = np.append(line_array_x, 255)
            
        else:
            line_array_x = np.append(line_array_x,np.sum(list_of_white_round_img[0][y:y+3,x:x+3] * x_line_matrix))
            
line_array_y = line_array_y.reshape(list_of_gray_img[0].shape)
line_array_x = line_array_x.reshape(list_of_gray_img[0].shape)
line_array_yx = line_array_y + line_array_x

cv2.imwrite("/home/dr169/Desktop/task/line-detect/Y-" + os.listdir("/home/dr169/Desktop/task/white-round/")[0], line_array_y)
cv2.imwrite("/home/dr169/Desktop/task/line-detect/X-" + os.listdir("/home/dr169/Desktop/task/white-round/")[0], line_array_x)
cv2.imwrite("/home/dr169/Desktop/task/line-detect/YX-" + os.listdir("/home/dr169/Desktop/task/white-round/")[0], line_array_yx)