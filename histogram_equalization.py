import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

im1 = cv.imread(r'C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images/shells.tif', cv.IMREAD_GRAYSCALE)
assert im1 is not None

h = cv.calcHist([im1], [0], None, [256], [0,256])
plt.bar(range(256), h.ravel())
plt.show()

im2 = cv.equalizeHist(im1)

h = cv.calcHist([im2], [0], None, [256], [0,256])
plt.bar(range(256), h.ravel())
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].imshow(im1, cmap='gray', vmin=0, vmax=255)
ax[1].imshow(im2, cmap='gray', vmin=0, vmax=255)
plt.show()