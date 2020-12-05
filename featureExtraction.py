import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats as stats
import math
import cv2
import itertools

class EdgeHistogramComputer:

    def __init__(self, rows, cols):
        sqrt2 = math.sqrt(2)
        self.kernels = (np.array([[1, 1], [-1, -1]]),  # Vertical Edge
                        np.array([[1, -1], [1, -1]]),  # Horizontal edge
                        np.array([[sqrt2, 0], [0, -sqrt2]]),  # Diagonal (45)
                        np.array([[0, sqrt2], [-sqrt2, 0]]),  # diagaonal (135)
                        np.array([[2, -2], [-2, 2]]))  # Non-Orientation
        self.bins = [len(self.kernels)]
        self.range = [0, len(self.kernels)]
        self.rows = rows
        self.cols = cols
        self.prefix = "EDH"

    def compute(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        descriptor = []
        dominantGradients = np.zeros_like(frame)
        maxGradient = cv2.filter2D(frame, cv2.CV_32F, self.kernels[0])
        maxGradient = np.absolute(maxGradient)
        for k in range(1, len(self.kernels)):
            kernel = self.kernels[k]
            gradient = cv2.filter2D(frame, cv2.CV_32F, kernel)
            gradient = np.absolute(gradient)
            np.maximum(maxGradient, gradient, maxGradient)
            indices = (maxGradient == gradient)
            dominantGradients[indices] = k

        frameH, frameW = frame.shape
        for row in range(self.rows):
            for col in range(self.cols):
                mask = np.zeros_like(frame)
                mask[int(((frameH / self.rows) * row)):int(((frameH / self.rows) * (row + 1))),
                int((frameW / self.cols) * col):int(((frameW / self.cols) * (col + 1)))] = 255
                hist = cv2.calcHist([dominantGradients], [0], mask, self.bins, self.range)
                hist = cv2.normalize(hist, None)
                descriptor.append(hist)

        # return np.concatenate([x for x in descriptor])
        descriptor = np.array(descriptor)
        globalEdges = np.transpose(descriptor.mean(0))[..., None]
        descriptor = np.append(descriptor, globalEdges, axis=0)
        descriptor = np.squeeze(descriptor, 2)
        return descriptor

# Feature Extraction
Images = np.load("Test_Images_32.npy")
Labels = np.load("Test_Labels_32.npy")
Images = np.float32(Images)
Labels = np.float32(Labels)
EHDComp = EdgeHistogramComputer(4, 4)

# Features is a row x 5 matrix, where the row corresponds to the image, col 1 is VE, col2 is HE, col3, is D45
# col 4 is D135, and col5 is NO. The data points are averages taken accross the entire image
Features_temp = []

# the data is still a little messed up due to the scaling with data max, though i dont think it should effect edges
for data in Images:
    info = np.finfo(data.dtype)
    data = data / data.max() # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)
    IMG_EHD = EHDComp.compute(img)
    Features_temp.append(IMG_EHD[16])

Features_temp = np.array(Features_temp)
Features = []

# split data into testing data and training data

for i in range(np.size(Labels)):  # 0-121 for this data set
    Features.append(Features_temp[i])

Features = np.array(Features)
Labels = np.array(Labels)

np.save('Test_Images_Features', Features)
np.save('Test_Labels_Features', Labels)

print("Features Vector of One Image: ")

print(Features)
print(Features.shape)
print(Labels.shape)