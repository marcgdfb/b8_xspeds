import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import itertools
import torch
from torch.nn import AvgPool2d,MaxPool2d
import cv2

list_darkData = [0,3,5,9,10,12,13,15,18]


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
    def printImage(image_data_matrix, index):
        plt.imshow(image_data_matrix, cmap="hot")
        plt.colorbar(label="Intensity")
        plt.title(f"Image {index}")
        plt.show()

    @staticmethod
    def printIntenstiy_horizontally(image_data_matrix, index):

        imMat = image_data_matrix

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
    def printIntenstiy_vertically(image_data_matrix, index):

        imMat = image_data_matrix

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
    def printIntenstiy_verticallyWithOutliers(image_data_matrix, index):

        imMat = image_data_matrix

        col_sums = np.sum(imMat, axis=0)  # Sum along each column
        col_indices = np.arange(len(col_sums))

        plt.figure(figsize=(8, 5))
        plt.scatter(col_indices, col_sums, marker='.',color='b', label='Column Sum',s=5)
        plt.xlabel("Column Index")
        plt.ylabel("Total Sum")
        plt.title(f"Column Sum for image {index}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()



    @staticmethod
    def histogram(image_data_matrix, index,bins=150,logarithmic=True):
        plt.hist(image_data_matrix.flatten(), bins)
        if logarithmic:
            plt.yscale('log')
        plt.title(f"Image {index} Raw Hist")
        plt.show()



    @staticmethod
    def viewAll():

        imageData=loadData()

        for num in range(10, len(imageData)):
            print(num)
            image_data_matrix_num = imageData[num]
            Investigate.printImage(image_data_matrix_num,num)
            Investigate.printIntenstiy_horizontally(image_data_matrix_num,num)
            Investigate.printIntenstiy_vertically(image_data_matrix_num,num)
            Investigate.histogram(image_data_matrix_num,num)

    @staticmethod
    def RemoveDarkImage(imageMat,darkImageMat,diagnostics=False):

        # Note if stored in uint16 (unsigned integers) a negative makes the values explode hence:
        removedDarkImageMat = imageMat.astype(np.int16) - darkImageMat.astype(np.int16)
        if diagnostics:
            print("Shape",imageMat.shape)
            print("maxVal",np.max(imageMat))
            print("minVal",np.min(imageMat))
            print("darkShape",darkImageMat.shape)
            print("darkMaxVal",np.max(darkImageMat))
            print("darkMinVal",np.min(darkImageMat))
            print("removedDarkShape",removedDarkImageMat.shape)
            print("removedDarkMaxVal",np.max(removedDarkImageMat))
            print("removedDarkMinVal",np.min(removedDarkImageMat))

        return np.where(removedDarkImageMat > 0,removedDarkImageMat,0)

    def darkImageRemoval(self,image_data_matrix,index,listDarkData,bins=150):

        for darknum in listDarkData:
            print("The Dark image is {}".format(darknum))
            darkMat = loadData()[darknum]



            dark_image_data_matrix = self.RemoveDarkImage(image_data_matrix,darkMat)

            plt.imshow(dark_image_data_matrix, cmap="hot")
            plt.colorbar(label="Intensity")
            plt.title(f"Image {index} with Dark Image {darknum} Removed")
            plt.show()

            plt.hist(dark_image_data_matrix.flatten(), bins)
            plt.yscale('log')
            plt.title(f"Image {index} with Dark Image {darknum} Removed Hist")
            plt.show()

    @staticmethod
    def plotMatClear(ImageMat,title,logarithmic=False):

        fig, ax = plt.subplots(figsize=(8, 8))
        # Get indices of nonzero elements
        y, x = np.nonzero(ImageMat)
        values = ImageMat[y, x]  # Get intensity values for color mapping

        scatter = ax.scatter(x, y, c=values, cmap='plasma', s=5, edgecolors='white', linewidth=0.2)
        plt.colorbar(scatter, ax=ax, label="Intensity")
        # Invert y-axis to match image orientation
        ax.set_ylim(ax.get_ylim()[::-1])

        # Set xticks and yticks to correspond to matrix indices
        ax.set_xticks(np.arange(0, ImageMat.shape[1], 250))
        ax.set_yticks(np.arange(0, ImageMat.shape[0], 250))

        # ax.set_xticklabels([int(i) for i in np.linspace(0, ImageMat.shape[1], 10)])  # Ensure integer labels
        # ax.set_yticklabels([int(i) for i in np.linspace(0, ImageMat.shape[0], 10)])

        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")
        if logarithmic:
            plt.yscale('log')
        ax.set_title(title)
        ax.set_aspect('equal')  # Ensure square pixels
        plt.show()

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
high_intensity_points = Clean.matrixAboveThreshold(array8Test,100)
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


imTest = high_intensity_points
imClear = np.asarray(Convolve(imTest,ktuple).avgPool())

imTestBinary = Clean.createBinaryMatrix(imTest,0)



if __name__ == '__main__':

    def plotRawMatThresholdedMat():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(array8Test, cmap='hot'), plt.title(f"Image 8 Raw")
        plt.subplot(1, 2, 2), plt.imshow(high_intensity_points, cmap='hot'), plt.title(f"Image 8 Thresholded above 100")
        plt.show()

    # Investigate.plotMatClear(Clean.matrixAboveThreshold(imData[9],100),"Image 9 thresholded above 100")

    # Investigate().histogram(imData[8],8,150,logarithmic=False)
    # plt.imshow(Clean.matrixAboveThreshold(array8Test,90), cmap="hot"), plt.title(f"image 8 thresholded above 90"), plt.show()
    plt.imshow(Clean.matrixAboveThreshold(imData[18], 90), cmap="hot"), plt.title(f"image 18 thresholded above 90"), plt.show()
    # plt.imshow(imVeryClear, cmap="hot"), plt.title(f"imVeryClear"), plt.show()
    # plt.imshow(doubledThresholdedPoints, cmap="hot"), plt.title(f"Thresholded Image 8 (200>I_pixel>100"), plt.show()
    # plt.imshow(imTensorThresholded),plt.title(f"Pooled Image {ktuple} greater than {lbThreshold} less than {ubThreshold}"),plt.show()


    pass
