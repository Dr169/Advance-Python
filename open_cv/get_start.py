import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/dr169/Pictures/IMG_20221219_135959_430.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./Codes/Tasks/OpenCV/images/output.jpg', img)

img_2 = cv2.imread('/home/dr169/Pictures/187394-full_rick-and-morty-wallpaper-iphone-rick-e-morty.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(img_2, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([])
plt.yticks([])
plt.plot([200,300,400],[100,200,300], 'c', linewidth=1)
plt.show()