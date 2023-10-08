import cv2

img = cv2.imread('/home/dr169/Pictures/IMG_20221219_135959_430.jpg',cv2.IMREAD_COLOR)

img[:,:50] = [255,255,255]
img[:,-50:] = [255,255,255]
img[:50,:] = [255,255,255]
img[-50:,:] = [255,255,255]

face = img[100:300,300:500]
img[50:250,50:250] = face

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()