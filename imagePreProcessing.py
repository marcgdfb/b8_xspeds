import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import itertools
import torch
from torch.nn import AvgPool2d,MaxPool2d
import cv2


def loadData():
    # The following code was provided by Sam Vinko, University of Oxford:

    # Name of the hdf file that contains the data we need
    f_name = 'sxro6416-r0504.h5'
    # Open the hdf5 file, use the path to the images to extract the data and place
    # it in the image data object for further manipulation and inspection.
    datafile = h5py.File(f_name, 'r')
    image_data = []
    for i in itertools.count(start=0):
        d = datafile.get(
            f'Configure:0000/Run:0000/CalibCycle:{i:04d}/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data')
        if d is not None:
            # actual image is at first index
            image_data.append(d[0])
        else:
            break
    # Tell me how many images were contained in the datafile
    # print(f"Loaded {len(image_data)} images.")

    # End of code provided

    return image_data

class Investigate:

    @staticmethod
    def printImage(image_data, index):
        plt.imshow(image_data[index], cmap="hot")
        plt.colorbar(label="Intensity")
        plt.title(f"Image {index}")
        plt.show()

    @staticmethod
    def printIntenstiy_horizontally(image_data, index):

        imMat = image_data[index]

        row_sums = np.sum(imMat, axis=1)  # Sum along each row
        row_indices = -np.arange(len(row_sums))

        plt.figure(figsize=(8, 5))
        plt.scatter(row_sums,row_indices, marker='.', color='b', label='Row Sum with - the index',s=5)
        plt.ylabel("Row Index")
        plt.xlabel("Total Sum")
        plt.title(f"Row Sum for image {index}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()


    @staticmethod
    def printIntenstiy_vertically(image_data, index):

        imMat = image_data[index]

        def removeOutliers():
            column_sums = np.sum(imMat, axis=0)  # Sum along each row
            column_indices = np.arange(len(column_sums))  # Row indices

            # Calculate IQR to find outliers
            Q1 = np.percentile(column_sums, 25)
            Q3 = np.percentile(column_sums, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outlier indices
            outlier_indices = np.where((column_sums < lower_bound) | (column_sums > upper_bound))[0]

            # Remove outliers from both arrays
            column_sums_noOutliers = np.delete(column_sums, outlier_indices)
            column_indices_noOutliers = np.delete(column_indices, outlier_indices)

            return column_sums_noOutliers,column_indices_noOutliers

        colsums_noOutliers, colIndices_noOutliers = removeOutliers()

        plt.figure(figsize=(8, 5))
        plt.scatter(colIndices_noOutliers, colsums_noOutliers, marker='.',color='b', label='Column Sum',s=5)
        plt.xlabel("Column Index")
        plt.ylabel("Total Sum")
        plt.title(f"Column Sum for image {index}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()


    @staticmethod
    def histogram(image_data, index,bins=150):
        plt.hist(image_data[index].flatten(), bins)
        plt.yscale('log')
        plt.title(f"Image {index} Raw Hist")
        plt.show()



    @staticmethod
    def viewAll():

        imageData=loadData()

        for num in range(10, len(imageData)):
            print(num)
            Investigate.printImage(imageData,num)
            Investigate.printIntenstiy_horizontally(imageData,num)
            Investigate.printIntenstiy_vertically(imageData,num)
            Investigate.histogram(imageData,num)

class Clean:

    @staticmethod
    def cleanNonCuveDataManual(imageMatrix, colLower=120, colLHigher=1550):
        """
        To prepare the matrix for fiting the lines, we manually
        remove the points either side of the curve
        e.g. imageMatrix=8 colLower=1236 colHigher=1550
        """
        # Without this following step the function permanently alters the input
        imageMatrix_copy = np.copy(imageMatrix)

        imageMatrix_copy[:, :colLower] = 0
        imageMatrix_copy[:, colLHigher:] = 0

        # Potentially remove edges
        # imageMatrix_copy[:200, :] = 0

        return imageMatrix_copy

    @staticmethod
    def removeTopData(imageMatrix, rowHigher=400):
        # Without this following step the function permanently alters the input
        imageMatrix_copy = np.copy(imageMatrix)

        imageMatrix_copy[:200, :] = 0

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
        print("Q1", Q1)
        Q3 = np.percentile(non_zero_mat_values, 75)
        print("Q3", Q3)
        return Q1, Q3

    @staticmethod
    def createNormalisedLogMatrix(imMatrix):

        imMatrix_copy = np.copy(imMatrix)

        imMatrix_normalised = imMatrix_copy / np.max(imMatrix_copy)

        log_imMatrix = np.log(imMatrix_normalised+1)

        return log_imMatrix

    @staticmethod
    def createNormalisedMatrix(imMatrix):
        imMatrix_copy = np.copy(imMatrix)
        imMatrix_normalised = imMatrix_copy / np.max(imMatrix_copy)
        return imMatrix_normalised


class Convolve:
    def __init__(self,imageMatrix,kernelSizeTuple):
        self.imageMatrix = imageMatrix
        self.kernelSizeTuple=kernelSizeTuple
        self.paddingTuple = (int(int(self.kernelSizeTuple[0] - 1) / 2), int(int(self.kernelSizeTuple[1] - 1) / 2))

    def avgPool(self):

        pool = AvgPool2d(kernel_size=self.kernelSizeTuple,
                         padding=self.paddingTuple, stride=1)
        return pool(torch.tensor(self.imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)

    def maxPool(self,):
        pool = MaxPool2d(kernel_size=self.kernelSizeTuple,
                         padding=self.paddingTuple, stride=1)
        return pool(torch.tensor(self.imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)

imData = loadData()

num = 8
array8Test = imData[8]
high_intensity_points = Clean.matrixAboveThreshold(array8Test,107)
high_intensity_points_binary = Clean.createBinaryMatrix(array8Test,107)
doubledThresholdedPoints = Clean.createNormalisedLogMatrix(np.where((array8Test > 100) & (array8Test < 200),array8Test,0))

doubledThresholdedPointsBinary = Clean.createBinaryMatrix(doubledThresholdedPoints,0)
# plt.imshow(doubledThresholdedPointsBinary,cmap="hot"),plt.title(f"Thresholded Image 8 (200>I_pixel>100) Binary"),plt.show()

# Try with removed top
# doubledThresholdedPoints = Clean.removeTopData(doubledThresholdedPoints,rowHigher=400)

ktuple = (11,3)
lbThreshold = 10
ubThreshold = 25

imTensor = np.asarray(Convolve(high_intensity_points,ktuple).avgPool())
imTensorThresholded = np.where((imTensor > 5) & (imTensor < 30),imTensor,0)

imVeryClear = np.asarray(Convolve(high_intensity_points,(21,5)).avgPool())


imTest = doubledThresholdedPoints
imClear = np.asarray(Convolve(imTest,ktuple).avgPool())

imTestBinary = Clean.createBinaryMatrix(imTest,0)



if __name__ == '__main__':
    plt.imshow(imVeryClear, cmap="hot"), plt.title(f"imVeryClear"), plt.show()
    # plt.imshow(doubledThresholdedPoints, cmap="hot"), plt.title(f"Thresholded Image 8 (200>I_pixel>100"), plt.show()
    # plt.imshow(imTensorThresholded),plt.title(f"Pooled Image {ktuple} greater than {lbThreshold} less than {ubThreshold}"),plt.show()

    pass