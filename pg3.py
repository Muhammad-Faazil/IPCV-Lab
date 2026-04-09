import cv2
import numpy as np
from skimage.feature import local_binary_pattern

import matplotlib.pyplot as plt

# Step 1: Read image
image = cv2.imread(&quot; &quot;)
# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Step 3: Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)
# Step 4: Sobel Edge Detection
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)
# Normalize sobel for better visualization (0 to 255)
sobel_norm = cv2.normalize(sobel_mag, None, 0, 255 cv2.NORM_MINMAX)
.astype(np.uint8)
# Step 5: Apply LBP
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(blur, n_points, radius, method=&quot;uniform&quot;)
# Display results
plt.figure(figsize=(12, 8))
plt.subplot(2,2,1)
plt.title(&quot;1. Original Grayscale&quot;)
plt.imshow(gray, cmap=&#39;gray&#39;)
plt.axis(&#39;off&#39;)

plt.subplot(2,2,2)
plt.title(&quot;2. Gaussian Blur (Noise Reduction)&quot;)
plt.imshow(blur, cmap=&#39;gray&#39;)

plt.axis(&#39;off&#39;)

plt.subplot(2,2,3)
plt.title(&quot;3. Sobel Edge Magnitude&quot;)
plt.imshow(sobel_norm, cmap=&#39;gray&#39;)
plt.axis(&#39;off&#39;)

plt.subplot(2,2,4)
plt.title(&quot;4. LBP Texture Features&quot;)
plt.imshow(lbp, cmap=&#39;gray&#39;)
plt.axis(&#39;off&#39;)

plt.tight_layout()
plt.show()
