import numpy as np
import cv2

def zoom_image(image, scale_factor):
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in the range (0, 10]")
    
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Create a new empty image to hold the zoomed version
    zoomed_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Nearest neighbor interpolation
    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i / scale_factor)
            orig_j = int(j / scale_factor)
            zoomed_image[i, j] = image[orig_i, orig_j]
    
    return zoomed_image

if __name__ == "__main__":
    # Load an image
    image = cv2.imread(r"C:\KDU\5 sem\Image Processing\Code\Assignments\assignment_01_images\a1q5images\im01small.png")
    
    # Define the zoom factor
    zoom_factor = 2.0
    
    # Zoom the image
    zoomed_image = zoom_image(image, zoom_factor)
    
    # Display the original and zoomed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Zoomed Image", zoomed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
