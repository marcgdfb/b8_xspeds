import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import itertools
import torch
from torch.nn import AvgPool2d,MaxPool2d
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

class Investigate:

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
    def cleanNonCuveDataManual(imageMatrix, colLower=120,colLHigher=1550):
        """
        To prepare the matrix for fiting the lines, we manually
        remove the points either side of the curve
        e.g imageMatrix=8 colLower=1236 colHigher=1550
        """
        # Without this following step the function permanently alters the input
        imageMatrix_copy = np.copy(imageMatrix)

        imageMatrix_copy[:, :colLower] = 0
        imageMatrix_copy[:, colLHigher:] = 0

        # Potentially remove edges
        # imageMatrix_copy[:200, :] = 0
        
        return imageMatrix_copy

    @staticmethod
    def createBinaryMatrix(imageMatrix, threshold):
        return np.where(imageMatrix > threshold, 1, 0)

    @staticmethod
    def matrixAboveThreshold(imageMatrix, threshold):
        return np.where(imageMatrix > threshold, imageMatrix, 0)

    @staticmethod
    def quartiles(imMatrix):
        mat_values = np.array(imMatrix).flatten()
        non_zero_mat_values = mat_values[mat_values != 0]
        if len(non_zero_mat_values) == 0:
            return None, None
        Q1 = np.percentile(non_zero_mat_values, 25)
        print("Q1",Q1)
        Q3 = np.percentile(non_zero_mat_values, 75)
        print("Q3",Q3)
        return Q1, Q3



def convolvedImage(imageMatrix,kernelSizeTuple=(5,3)):

    paddingTuple = (int(int(kernelSizeTuple[0]-1)/2),int(int(kernelSizeTuple[1]-1)/2))

    pool = AvgPool2d(kernel_size=kernelSizeTuple,
                     padding=paddingTuple, stride=1)
    return pool(torch.tensor(imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)

def convolvedImageMaxPool(imageMatrix,kernelSizeTuple=(5,3),paddingTuple=(2,1)):

    pool = MaxPool2d(kernel_size=kernelSizeTuple,
                     padding=paddingTuple, stride=1)
    return pool(torch.tensor(imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)



num = 8
array8Test = image_data[8]
high_intensity_points = Clean.matrixAboveThreshold(array8Test,107)
doubledThresholdedPoints = np.where((array8Test > 100) & (array8Test < 200),array8Test,0)
# plt.imshow(doubledThresholdedPoints,cmap="hot"),plt.title(f"Thresholded Image 8 (200>I_pixel>100"),plt.show()
high_intensity_points_binary = Clean.createBinaryMatrix(array8Test,100)

"""
imTensor31 = np.asarray(convolvedImageMaxPool(high_intensity_points,kernelSizeTuple=(3,1),paddingTuple=(1,0)))
imTensor51 = np.asarray(convolvedImageMaxPool(high_intensity_points,kernelSizeTuple=(5,1),paddingTuple=(2,0)))

im31 = np.where((imTensor31 > 100) & (imTensor31 < 200),imTensor31,0)
im51 = np.where((imTensor51 > 100) & (imTensor51 < 200),imTensor31,0)

imTensor31Avg31 = convolvedImage(im31, kernelSizeTuple=(3, 1), paddingTuple=(1, 0))
imTensor51Avg31 = convolvedImage(im51, kernelSizeTuple=(3, 1), paddingTuple=(1, 0))

imTmax31avg31avg113 = convolvedImage(imTensor31Avg31, kernelSizeTuple=(11, 3), paddingTuple=(5, 1))
imTmax51avg31avg113 = convolvedImage(imTensor51Avg31, kernelSizeTuple=(11, 3), paddingTuple=(5, 1))

im31_2 = np.where((imTmax31avg31avg113 > 15) & (imTmax31avg31avg113 < 200), imTmax31avg31avg113, 0)
im51_2 = np.where((imTmax51avg31avg113 > 15) & (imTmax51avg31avg113 < 200), imTmax51avg31avg113, 0)

imTensorMaxAvg73 = np.asarray(convolvedImage(convolvedImageMaxPool(high_intensity_points,kernelSizeTuple=(7,3),paddingTuple=(3,1)),
                                             kernelSizeTuple=(7,5),paddingTuple=(3,2)))
imTensorMaxAvg73Greaterthan80 = np.where((imTensorMaxAvg73 > 80) & (imTensorMaxAvg73 < 200), imTensorMaxAvg73, 0)
binaryImTensorMaxAvg73Cleaned = (Clean.createBinaryMatrix(
          Clean.cleanNonCuveDataManual(imTensorMaxAvg73Greaterthan80,1200,1550),threshold=0))

imTensorMax33Avg73 = np.asarray(convolvedImage(convolvedImageMaxPool(high_intensity_points,kernelSizeTuple=(3,3),paddingTuple=(1,1)),
                                             kernelSizeTuple=(7,5),paddingTuple=(3,2)))
imTensorMax33Avg73Greaterthan = np.where((imTensorMax33Avg73 > 40) & (imTensorMax33Avg73 < 2000), imTensorMax33Avg73, 0)

ImTensorMax33Avg73Cleaned = Clean.cleanNonCuveDataManual(imTensorMax33Avg73Greaterthan,1235,1550)

imTmax31avg31avg213 = convolvedImage(imTensor31Avg31, kernelSizeTuple=(21, 3), paddingTuple=(10, 1))
imTmax31avg31avg213_th = np.where((imTmax31avg31avg213 > 10), imTmax31avg31avg213, 0)
"""

ktuple = (11,3)
lbThreshold = 5
ubThreshold = 25

imTensor = np.asarray(convolvedImage(doubledThresholdedPoints,kernelSizeTuple=ktuple))
imTensorThresholded = np.where((imTensor > lbThreshold) & (imTensor < ubThreshold),imTensor,0)

imTest = imTensorThresholded
plt.imshow(imTest),plt.title(f"Pooled Image {ktuple} greater than {lbThreshold} less than {ubThreshold}"),plt.show()

# Apply Gaussian blur to reduce noise
# blurred_img = cv2.GaussianBlur(doubledThresholdedPoints, (5, 5), sigmaX=1.4)
# plt.imshow(blurred_img),plt.title(f"doubledThresholdedPoints Image {ktuple} with Gaussian Blur"),plt.show()


# Apply Canny Edge Detector
# edges = cv2.Canny(np.uint8(imTest), 100, 200)
# # Display results
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1), plt.imshow(imTest, cmap='hot'), plt.title(f"Pooled Image {ktuple}")
# plt.subplot(1,2,2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
# plt.show()

# # Apply Canny Edge Detector
# edges = cv2.Canny(np.uint8(doubledThresholdedPoints), 100, 200)
# edgesSideRemoved = Clean.cleanNonCuveDataManual(edges,1236,1550)
#
# # Display results
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1), plt.imshow(high_intensity_points, cmap='hot'), plt.title('Original Image')
# plt.subplot(1,2,2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
# plt.show()




# def preProcessForFit():
#
#     # plt.imshow(high_intensity_points, cmap='hot'), plt.title('Original Image thresholded')
#
#     # plt.figure(figsize=(10,5))
#     # plt.subplot(1,2,1), plt.imshow(np.where((imTensor31 > 100) & (imTensor31 < 200),imTensor31,0), cmap='hot'), plt.title('31')
#     # plt.subplot(1,2,2), plt.imshow(np.where((imTensor51 > 100) & (imTensor51 < 200),imTensor51,0), cmap='hot'), plt.title('51')
#     # plt.show()
#
#     imTmax31avg31avg73 = convolvedImage(imTensor31Avg31,kernelSizeTuple=(7,3),paddingTuple=(3,1))
#     imTmax51avg31avg73 = convolvedImage(imTensor51Avg31,kernelSizeTuple=(7,3),paddingTuple=(3,1))
#
#     im31 = np.where((imTensor31 > 20) & (imTensor31 < 200),imTensor31,0)
#     im51 = np.where((imTensor51 > 20) & (imTensor51 < 200),imTensor31,0)
#
#     # plt.figure(figsize=(10,5))
#     # plt.subplot(1,2,1), plt.imshow(imTmax31avg31avg73, cmap='hot'), plt.title('imTmax31avg31avg73')
#     # plt.subplot(1,2,2), plt.imshow(imTmax51avg31avg73, cmap='hot'), plt.title('imTmax51avg31avg73')
#     # plt.show()
#     #
#     # plt.figure(figsize=(10,5))
#     # plt.subplot(1,2,1), plt.imshow(im31, cmap='hot'), plt.title('imTmax31avg31avg73 greater than 20')
#     # plt.subplot(1,2,2), plt.imshow(im51, cmap='hot'), plt.title('imTmax51avg31avg73 greater than 20')
#     # plt.show()
#
#
#
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1), plt.imshow(imTmax31avg31avg113, cmap='hot'), plt.title('imTmax31avg31avg113')
#     # plt.subplot(1, 2, 2), plt.imshow(imTmax51avg31avg113, cmap='hot'), plt.title('imTmax51avg31avg113')
#     # plt.show()
#
#
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1), plt.imshow(im31, cmap='hot'), plt.title('imTmax31avg31avg113 greater than 15')
#     # plt.subplot(1, 2, 2), plt.imshow(im51, cmap='hot'), plt.title('imTmax51avg31avg113 greater than 15')
#     # plt.show()



# # imTensor = convolvedImageMaxPool(high_intensity_points,kernelSizeTuple=(11,3),paddingTuple=(5,1))
# # covTest = np.asarray(imTensor)
# # covTestThreshold = np.where(covTest>100,covTest,0)
# #
# # # covTest2 = convolvedImage(covTestThreshold,kernelSizeTuple=(5,3),paddingTuple=(2,1))
# # covTest2_inter = convolvedImage(covTestThreshold,kernelSizeTuple=(3,1),paddingTuple=(1,0))
# # covTest2 = convolvedImage(covTestThreshold,kernelSizeTuple=(3,3),paddingTuple=(1,1))
# # covTest2Threshold = np.where(covTest2>80,covTest,0)
# #
# # edgescovTest1 = cv2.Canny(np.uint8(covTestThreshold), 100, 200)
# # edgescovTest2 = cv2.Canny(np.uint8(covTest2Threshold), 100, 200)
# # print(covTest2[1396,1442])


if __name__ == "__main__":
    pass
    # preProcessForFit()