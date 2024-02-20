import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv.imread(r'C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images\daisy.jpg', cv.IMREAD_COLOR)
assert image is not None

# Create a mask where everything is initialized to background
mask = np.zeros(image.shape[:2], np.uint8)

# Define the rectangle containing the foreground object
rect = (30,150, 539, 400)  # (x, y, w, h)

# Run GrabCut algorithm
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
cv.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)

# Create mask where the foreground and possible foreground are labeled as 1
foreground_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
background_mask = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')
# Apply the mask to the original image
foreground_image = image * foreground_mask[:, :, np.newaxis]

# Apply inverse mask to get the background

background_image = image * background_mask[:, :, np.newaxis]

blured_background=cv.blur(src=background_image,ksize=(50,50))
enhanced_image=cv.add(foreground_image,blured_background)
#cv.imshow(blured_background)
# Show the images using ax
fig, ax = plt.subplots(1, 5, figsize=(10, 10))

# Segmentation Mask
ax[0].imshow(foreground_mask, cmap='gray')
ax[0].set_title('Segmentation Mask')

# Foreground Image
ax[1].imshow(cv.cvtColor(foreground_image, cv.COLOR_BGR2RGB))
ax[1].set_title('Foreground Image')

# Background Image
ax[2].imshow(cv.cvtColor(background_image, cv.COLOR_BGR2RGB))
ax[2].set_title('Background Image')

ax[3].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
ax[3].set_title('Original image')

ax[4].imshow(cv.cvtColor(enhanced_image, cv.COLOR_BGR2RGB))
ax[4].set_title('enhanced image')

plt.show()
