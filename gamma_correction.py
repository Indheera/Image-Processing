import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

img_orig = cv.imread(r'C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images\highlights_and_shadows.jpg', cv.IMREAD_COLOR)
assert img_orig is not None

lab_image = cv.cvtColor(img_orig, cv.COLOR_BGR2LAB)

l, a, b = cv.split(lab_image)

gamma = 0.2
table = np.array([(i/255)**(gamma)*255 for i in np.arange(0,256)]).astype('uint8')
img_gamma = cv.LUT(l, table)

img_corrected = cv.merge((img_gamma,a,b))
img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

img_corrected = cv.cvtColor(img_corrected, cv.COLOR_LAB2RGB)
f, axarr = plt.subplots(3,2)
axarr[0,0].imshow(img_orig)
axarr[0,1].imshow(img_corrected)
color = ('b', 'g', 'r')

for i, c in enumerate(color):
    hist_orig = cv.calcHist([img_orig], [i], None, [256], [0,256])
    axarr[1,0].plot(hist_orig, color = c)
    hist_gamma = cv.calcHist([img_corrected], [i], None, [256], [0,256])
    axarr[1,1].plot(hist_gamma, color = c)

axarr[2,0].plot(table)
axarr[2,0].set_xlim(0,255)
axarr[2,0].set_ylim(0,255)
axarr[2,0].set_aspect('equal')    

plt.show()
