import numpy as np
import matplotlib.pyplot as plt
import h5py
import itertools
import torch
from torch.nn import AvgPool2d
import cv2

# The following code was provided by Sam Vinko, University of Oxford:

# Name of the hdf file that contains the data we need
f_name = 'sxro6416-r0504.h5'
# Open the hdf5 file, use the path to the images to extract the data and place
# it in the image data object for further manipulation and inspection.
datafile = h5py.File(f_name, 'r')
image_data = []
for i in itertools.count(start=0):
    d = datafile.get(f'Configure:0000/Run:0000/CalibCycle:{i:04d}/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data')
    if d is not None:
        # actual image is at first index
        image_data.append(d[0])
    else:
        break
# Tell me how many images were contained in the datafile
print(f"Loaded {len(image_data)} images.")

# End of code provided

class Investigate():

    @staticmethod
    def showImage(index):
        # Plot a good dataset - here index #8 (but there are others too!)
        plt.imshow(image_data[index], cmap="cividis")
        plt.colorbar(label="Intensity")
        plt.title(f"Image {index}")
        # plt.savefig(f"Image Raw {index}")
        plt.show()

    @staticmethod
    def showHist(index,bins=100):
        # The histogram of the data will help show possible single photon hits
        plt.hist(image_data[index].flatten(), bins)
        plt.yscale('log')
        plt.title(f"Image {index} Raw Hist")
        # plt.savefig(f"Image {index} Raw Hist")
        plt.show()

    @staticmethod
    def showImageAndHist(index,bins=100):
        Investigate.showImage(index)
        Investigate.showHist(index,bins)


    @staticmethod
    def showImageIntensityHigherThan(index, intensity_min=100):
        # Plot a good dataset - here index #8 (but there are others too!)

        arr = image_data[index]
        high_intensity_points = np.where(arr > intensity_min, arr, np.nan)
        plt.imshow(high_intensity_points)
        plt.colorbar(label="Intensity")
        plt.title(f"Image {index}") 
        # plt.savefig(f"Image Raw {index}")
        plt.show()

        # The histogram of the data will help show possible single photon hits
        plt.hist(high_intensity_points.flatten(), bins=100)
        plt.yscale('log')
        plt.title(f"Image {index} Raw Hist")
        # plt.savefig(f"Image {index} Raw Hist")
        plt.show()

class Clean:

    @staticmethod
    def cleanNonCuveDataManual(imageMatrix, colLower,colLHigher):
        """
        To prepare the matrix for fiting the lines, we manually
        remove the points either side of the curve
        e.g imageMatrix=8 colLower=1236 colHigher=1550
        """
        # Without this following step the function permanently alters the input
        imageMatrix_copy = np.copy(imageMatrix)

        imageMatrix_copy[:, :colLower] = 0
        imageMatrix_copy[:, colLHigher:] = 0
        
        return imageMatrix_copy

    @staticmethod
    def createBinaryMatrix(imageMatrix, threshold):
        return np.where(imageMatrix > threshold, 1, 0)

    @staticmethod
    def matrixAboveThreshold(imageMatrix, threshold):
        return np.where(imageMatrix > threshold, imageMatrix, 0)



def investigateImageData(index):
    print(type(image_data[index]))
    # np.set_printoptions(threshold=np.inf)
    # print(image_data[index])
    arr = image_data[index]
    print(arr.shape)

def convolvedImage(imageMatrix):
    pool = AvgPool2d(kernel_size=(5,3), padding=(2,1), stride=1)
    return pool(torch.tensor(imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)

# for num in range(0,len(image_data)):
#     print(num)
#     Investigate.showImageAndHist(num)

num = 8
array8Test = image_data[8]
high_intensity_points = Clean.matrixAboveThreshold(array8Test,100)
high_intensity_points_binary = Clean.createBinaryMatrix(array8Test,100)

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1), plt.imshow(array8Test, cmap='hot'), plt.title('Original Image')
# plt.subplot(1,2,2), plt.imshow(high_intensity_points, cmap='hot'), plt.title('Thresholded')
# plt.show()

imTensor = convolvedImage(high_intensity_points)
covTest = np.asarray(imTensor)

reducedCovTest = Clean.createBinaryMatrix(
    Clean.cleanNonCuveDataManual(covTest,1236,1550),0)
# plt.imshow(reducedCovTest,cmap="hot")
# plt.show()


# plt.figure(figsize=(10, 10))
#
# plt.subplot(2, 2, 1)
# plt.imshow(array8Test, cmap='hot')
# plt.title('Original Image')
#
# plt.subplot(2, 2, 2)
# plt.imshow(high_intensity_points, cmap='hot')
# plt.title('Thresholded')
#
# plt.subplot(2, 2, 3)
# plt.imshow(covTest, cmap='hot')
# plt.title('covTest')
#
# plt.subplot(2, 2, 4)
# plt.imshow(reducedCovTest, cmap='hot')
# plt.title('reducedCovTest')
#
# plt.tight_layout()  # Ensures proper spacing between subplots
# plt.show()


# Apply Canny Edge Detector
# edges = cv2.Canny(np.uint8(high_intensity_points), 100, 200)
# edgesSideRemoved = Clean.cleanNonCuveDataManual(edges,1236,1550)
# print(sum(edges.flatten()))
# # Display results
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1), plt.imshow(high_intensity_points, cmap='hot'), plt.title('Original Image')
# plt.subplot(1,2,2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
# plt.show()


# Investigate.showImageAndHist(num,200)
# Investigate.showImageIntensityHigherThan(num,100)

