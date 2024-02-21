import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images\einstein.png', cv.IMREAD_GRAYSCALE)

sobel_x = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)], dtype ='float')
gradient_x = cv.filter2D(img, -1, sobel_x)

sobel_y = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)], dtype = 'float')
gradient_y = cv.filter2D(img, -1, sobel_y)

fig, ax = plt.subplots(1,3, sharex='all', sharey='all', figsize=(10,10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(gradient_x, cmap='gray')
ax[1].set_title('Gradient X')

ax[2].imshow(gradient_y, cmap='gray')
ax[2].set_title('gradient_y')
