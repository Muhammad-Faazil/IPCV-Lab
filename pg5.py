import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read input image from given path
img = cv2.imread(r"D:\DHWANI\ENGINEERING\VI\IPCV\ipcv_lab\image.jpg")

# Convert BGR image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to get binary image
_, otsu_thresholded_image = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Create 3x3 structuring element (kernel)
kernel = np.ones((3, 3))
kernel = kernel.astype(np.uint8)

# Perform erosion (shrinks white regions)
erosion = cv2.erode(otsu_thresholded_image, kernel)

# Extract inner edges (original - erosion)
edge = cv2.subtract(otsu_thresholded_image, erosion)

# Perform dilation (expands white regions)
dilation = cv2.dilate(otsu_thresholded_image, kernel)

# Extract outer edges (dilation - original)
sub_sub = cv2.subtract(dilation, otsu_thresholded_image)

# Create figure for displaying results
plt.figure(figsize=(10, 10))

# Display grayscale image
plt.subplot(3, 2, 1)
plt.title("Grayscale image")
plt.imshow(img_gray, cmap='gray')

# Display binary image
plt.subplot(3, 2, 2)
plt.title("Binary image")
plt.imshow(otsu_thresholded_image)

# Display eroded image
plt.subplot(3, 2, 3)
plt.title("Eroded image")
plt.imshow(erosion)

# Display dilated image
plt.subplot(3, 2, 4)
plt.title("Dilated image")
plt.imshow(dilation)

# Display outer edge result
plt.subplot(3, 2, 5)
plt.title("Resulting image")
plt.imshow(sub_sub)

# Adjust layout
plt.tight_layout()

# Show all plots
plt.show()