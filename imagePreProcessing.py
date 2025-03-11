import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import itertools
import torch
from torch.nn import AvgPool2d, MaxPool2d
import cv2
matplotlib.use('TkAgg')


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
        plt.scatter(row_sums, row_indices, marker='.', color='b', label='Row Sum with - the index', s=5)
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

            return column_sums_noOutliers, column_indices_noOutliers

        colsums_noOutliers, colIndices_noOutliers = removeOutliers()

        plt.figure(figsize=(8, 5))
        plt.scatter(colIndices_noOutliers, colsums_noOutliers, marker='.', color='b', label='Column Sum', s=5)
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
        plt.scatter(col_indices, col_sums, marker='.', color='b', label='Column Sum', s=5)
        plt.xlabel("Column Index")
        plt.ylabel("Total Sum")
        plt.title(f"Column Sum for image {index}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    @staticmethod
    def histogram(image_data_matrix, index, bins=150, logarithmic=True):
        plt.hist(image_data_matrix.flatten(), bins)
        if logarithmic:
            plt.yscale('log')
        plt.title(f"Image {index} Raw Hist")
        plt.show()

    @staticmethod
    def viewAll():

        imageData = loadData()

        for num in range(10, len(imageData)):
            print(num)
            image_data_matrix_num = imageData[num]
            Investigate.printImage(image_data_matrix_num, num)
            Investigate.printIntenstiy_horizontally(image_data_matrix_num, num)
            Investigate.printIntenstiy_vertically(image_data_matrix_num, num)
            Investigate.histogram(image_data_matrix_num, num)

    @staticmethod
    def RemoveDarkImage(imageMat, darkImageMat, diagnostics=False):

        # Note if stored in uint16 (unsigned integers) a negative makes the values explode hence:
        removedDarkImageMat = imageMat.astype(np.int16) - darkImageMat.astype(np.int16)
        if diagnostics:
            print("Shape", imageMat.shape)
            print("maxVal", np.max(imageMat))
            print("minVal", np.min(imageMat))
            print("darkShape", darkImageMat.shape)
            print("darkMaxVal", np.max(darkImageMat))
            print("darkMinVal", np.min(darkImageMat))
            print("removedDarkShape", removedDarkImageMat.shape)
            print("removedDarkMaxVal", np.max(removedDarkImageMat))
            print("removedDarkMinVal", np.min(removedDarkImageMat))

        return np.where(removedDarkImageMat > 0, removedDarkImageMat, 0)

    def darkImageRemoval(self, image_data_matrix, index, listDarkData, bins=150):

        for darknum in listDarkData:
            print("The Dark image is {}".format(darknum))
            darkMat = loadData()[darknum]

            dark_image_data_matrix = self.RemoveDarkImage(image_data_matrix, darkMat)

            plt.imshow(dark_image_data_matrix, cmap="hot")
            plt.colorbar(label="Intensity")
            plt.title(f"Image {index} with Dark Image {darknum} Removed")
            plt.show()

            plt.hist(dark_image_data_matrix.flatten(), bins)
            plt.yscale('log')
            plt.title(f"Image {index} with Dark Image {darknum} Removed Hist")
            plt.show()

    @staticmethod
    def plotMatClear(ImageMat, title, logarithmic=False,withLines=False):

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

        if withLines:
            y_vals = np.linspace(0, ImageMat.shape[0], 1000)

            # Compute x values for both parabolas
            A1 = 6.613794078473409e-05
            B1 = 862
            C1 = 1278
            A2 = 7.063102423636962e-05
            B2 = 862
            C2 = 1418

            x1_vals = A1 * (y_vals - B1) ** 2 + C1
            x2_vals = A2 * (y_vals - B2) ** 2 + C2

            ax.plot(x1_vals, y_vals, color='cyan', linewidth=1.5, linestyle='--', label='Beta Line')
            ax.plot(x2_vals, y_vals, color='magenta', linewidth=1.5, linestyle='--', label='Alpha Line')
            ax.legend()

        plt.show()

    @staticmethod
    def histogramsSubSquares(imageMatrix, iStart, iLength, jStart, jLength, title, bins=150, logarithmic=False,thresholdADU=0,
                             verticalLineVal=0):

        imSquare = imageMatrix[iStart:iStart + iLength, jStart:jStart + jLength]
        vals = imSquare.flatten()
        vals = vals[vals > thresholdADU]
        plt.hist(vals, bins)
        if logarithmic:
            plt.yscale('log')
        if verticalLineVal != 0:
            plt.axvline(x=verticalLineVal)
        plt.title(title)
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

        imageMatrix_copy[:rowHigher, :] = 0

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

        log_imMatrix = np.log(imMatrix_normalised + 1)

        return log_imMatrix

    @staticmethod
    def createNormalisedMatrix(imMatrix):
        imMatrix_copy = np.copy(imMatrix)
        imMatrix_normalised = imMatrix_copy / np.max(imMatrix_copy)
        return imMatrix_normalised


class Convolve:
    def __init__(self, imageMatrix, kernelSizeTuple):
        self.imageMatrix = imageMatrix
        self.kernelSizeTuple = kernelSizeTuple
        self.paddingTuple = (int(int(self.kernelSizeTuple[0] - 1) / 2), int(int(self.kernelSizeTuple[1] - 1) / 2))

    def avgPool(self):
        pool = AvgPool2d(kernel_size=self.kernelSizeTuple,
                         padding=self.paddingTuple, stride=1)
        return pool(torch.tensor(self.imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)

    def maxPool(self, ):
        pool = MaxPool2d(kernel_size=self.kernelSizeTuple,
                         padding=self.paddingTuple, stride=1)
        return pool(torch.tensor(self.imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)

def imVeryClear(matrixOfInterest,thr=100,ktuple=(11,3)):

    mat_thr = np.where(matrixOfInterest > thr, matrixOfInterest, 0)
    return np.asarray(Convolve(mat_thr, ktuple).avgPool())


imData = loadData()

list_darkNoPeaks = [3,5,9,12,13,15,18]
list_dark_smallPeaks = [0,10]
list_darkData = [0, 3, 5, 9, 10, 12, 13, 15, 18]

list_data = []
for idx_data in range(len(imData)):
    if idx_data not in list_darkData:
        list_data.append(idx_data)


array8Test = imData[8]
high_intensity_points = Clean.matrixAboveThreshold(array8Test, 100)
im_very_clear = np.asarray(Convolve(high_intensity_points, (21, 5)).avgPool())
imTest = high_intensity_points


imTestBinary = Clean.createBinaryMatrix(imTest, 0)

if __name__ == '__main__':
    def plotRawMatThresholdedMat(indexOfInterest=8):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(imData[indexOfInterest], cmap='hot'), plt.title(f"Image 8 Raw")
        thr = 90
        plt.subplot(1, 2, 2), plt.imshow(Clean.matrixAboveThreshold(imData[indexOfInterest], thr), cmap='hot'), plt.title(
            f"Image 8 Thresholded above {thr}")
        plt.show()


    plotRawMatThresholdedMat(1)

    def plotThreeDeviations(indexOfInterest=8,zoomArea=(500,1000,500,1000)):

        mean = 60
        sigma = 10
        imMat = imData[indexOfInterest]

        mat_minusMean = imMat.astype(np.int16) - mean
        mat_minusMean[mat_minusMean < 0] = 0

        imMat1Sigma = np.where(mat_minusMean > sigma, mat_minusMean, 0)[zoomArea[0]:zoomArea[1], zoomArea[2]:zoomArea[3]]
        imMat2Sigma = np.where(mat_minusMean > 2 * sigma, mat_minusMean, 0)[zoomArea[0]:zoomArea[1], zoomArea[2]:zoomArea[3]]
        imMat3Sigma = np.where(mat_minusMean > 3 * sigma, mat_minusMean, 0)[zoomArea[0]:zoomArea[1], zoomArea[2]:zoomArea[3]]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1), plt.imshow(imMat1Sigma, cmap='hot'), plt.title(f"Image {indexOfInterest} thresholded above 1 sigma")
        plt.subplot(1, 3, 2), plt.imshow(imMat2Sigma,cmap='hot'), plt.title(f"Image {indexOfInterest} thresholded above 2 sigma")
        plt.subplot(1, 3, 3), plt.imshow(imMat3Sigma, cmap='hot'), plt.title(f"Image {indexOfInterest} thresholded above 3 sigma")
        plt.show()

    # plotThreeDeviations(1,(500,600,500,600))

    # plotRawMatThresholdedMat(1)

    def plotRawDoubleThresholded(indexOfInterest=8,lb=90,ub=110):
        matrixOfInterest = imData[indexOfInterest]
        doub_thr_MoI = np.where((matrixOfInterest > lb) & (matrixOfInterest < ub), matrixOfInterest, 0)

        title = f"Image {indexOfInterest} plotted thresholdeded between {lb} and {ub}"
        plt.imshow(doub_thr_MoI)
        plt.title(title)
        plt.show()

    # plotRawDoubleThresholded(8,75,3000)



    def clearPlotInvestigations(indexOfInterest=8,lb=90,ub=0):

        matrixOfInterest = imData[indexOfInterest]

        if ub != 0:
            MoI = np.where((matrixOfInterest > lb) & (matrixOfInterest < ub), matrixOfInterest, 0)
            title = f"Image {indexOfInterest} plotted thresholdeded between {lb} and {ub}"
        else:
            MoI = np.where(matrixOfInterest > lb, matrixOfInterest, 0)
            title = f"Image {indexOfInterest} plotted thresholdeded above {lb}"

        Investigate.plotMatClear(MoI,title,withLines=True)





    def histograms(indexOfInterest=8,verticalLine=0):
        # Thresholded8Above90 = Clean.matrixAboveThreshold(array8Test, 80)
        # flattenedvals = Thresholded8Above90.flatten()
        # nonzeroVals = flattenedvals[flattenedvals != 0]
        # plt.hist(nonzeroVals, bins=100)
        # plt.yscale('log')
        # plt.title(f"Image 8 Thresholded above 90 Hist")
        # plt.show()


        iIndexStart = 500
        iIndexEnd = 1750
        jIndexStart = 50
        jIndexEnd = 1150

        titleH = f"Image {indexOfInterest} Histogram for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "

        Investigate().histogramsSubSquares(imageMatrix=imData[indexOfInterest],
                                           iStart=iIndexStart,
                                           iLength=iIndexEnd - iIndexStart,
                                           jStart=jIndexStart,
                                           jLength=jIndexEnd - jIndexStart,
                                           title=titleH,
                                           bins=200,
                                           logarithmic=True,
                                           thresholdADU=0,
                                           verticalLineVal=verticalLine
                                           )

    # histograms(0)


    # plt.imshow(Clean.matrixAboveThreshold(array8Test,90), cmap="hot"), plt.title(f"image 8 thresholded above 90"), plt.show()
    # plt.imshow(Clean.matrixAboveThreshold(imData[18], 90), cmap="hot"), plt.title(f"image 18 thresholded above 90"), plt.show()
    # plt.imshow(imVeryClear, cmap="hot"), plt.title(f"imVeryClear"), plt.show()
    # plt.imshow(doubledThresholdedPoints, cmap="hot"), plt.title(f"Thresholded Image 8 (200>I_pixel>100"), plt.show()
    # plt.imshow(imTensorThresholded),plt.title(f"Pooled Image {ktuple} greater than {lbThreshold} less than {ubThreshold}"),plt.show()

    # plt.imshow(im_very_clear, cmap='hot')
    # plt.show()


    # Investigate.printIntenstiy_vertically(high_intensity_points,"8")

    pass
