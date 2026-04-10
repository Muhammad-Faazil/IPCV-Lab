import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r"D:\DHWANI\ENGINEERING\VI\IPCV\ipcv_lab\image.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, otsu_thresholded_image = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

kernel = np.ones((3, 3))
kernel = kernel.astype(np.uint8)

erosion = cv2.erode(otsu_thresholded_image, kernel)

edge = cv2.subtract(otsu_thresholded_image, erosion)

dilation = cv2.dilate(otsu_thresholded_image, kernel)

sub_sub = cv2.subtract(dilation, otsu_thresholded_image)

plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.title("Grayscale image")
plt.imshow(img_gray, cmap='gray')

plt.subplot(3, 2, 2)
plt.title("Binary image")
plt.imshow(otsu_thresholded_image)

plt.subplot(3, 2, 3)
plt.title("Eroded image")
plt.imshow(erosion)

plt.subplot(3, 2, 4)
plt.title("Dilated image")
plt.imshow(dilation)

plt.subplot(3, 2, 5)
plt.title("Resulting image")
plt.imshow(sub_sub)

plt.tight_layout()
plt.show()
