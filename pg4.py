import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"D:\DHWANI\ENGINEERING\VI\IPCV\ipcv_lab\image.jpg")

if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image.copy()

low_contrast = cv2.convertScaleAbs(gray, alpha=0.5, beta=50)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(low_contrast)

blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

ret, segmented = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("1. Low Contrast Image")
plt.imshow(low_contrast, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("2. Enhanced (CLAHE)")
plt.imshow(enhanced, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("3. Segmented (Otsu)")
plt.imshow(segmented, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 9: Stop
