# 30 - 03 - 2026

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'D:\4SF23CI052 - IPCV\wall-39.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
_,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(binary, kernel, iterations=1)
erosion = cv2.erode(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16, 9))

plt.subplot(2, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('OG Gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary')
plt.axis('off')

plt.tight_layout()
plt.show()
