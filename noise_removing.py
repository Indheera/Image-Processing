import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread(r'C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images\rice_gaussian_noise.png')
assert im1 is not None

im2 = cv.imread(r'C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images\rice_salt_pepper_noise.png')
assert im2 is not None 

gaussian_blur = cv.GaussianBlur(im1, (7,7), 0)
median_blur = cv.medianBlur(im2, 5)

gray_image = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2GRAY)
ret, segmented_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = np.ones((1, 1), np.uint8)
processed_image = cv.morphologyEx(segmented_image, cv.MORPH_OPEN, kernel)
processed_image = cv.morphologyEx(segmented_image, cv.MORPH_CLOSE, kernel)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(processed_image, connectivity=4)
num_rice_grains = num_labels - 1

print("Number of rice grains:", num_rice_grains)

fig, ax = plt.subplots(3,2, figsize=(10,10))
ax[0,0].imshow(im1, vmin=0, vmax=255)
ax[0,0].set_title('Original Gaussian noised image')
ax[0,1].imshow(gaussian_blur, vmin=0, vmax=255)
ax[0,1].set_title('Gaussian Blured')
ax[1,0].imshow(im2, vmin=0, vmax=255)
ax[1,0].set_title('Original Salt-and-pepper noised Image')
ax[1,1].imshow(median_blur, vmin=0, vmax=255)
ax[1,1].set_title('Median Blured')
ax[2,0].imshow(segmented_image, vmin=0, vmax=255, cmap = 'gray')
ax[2,0].set_title('segmented_image')
ax[2,1].imshow(processed_image, cmap = 'gray')
ax[2,1].set_title('Segmented Image (Processed)')


