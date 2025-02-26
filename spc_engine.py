from imagePreProcessing import *
from scipy.optimize import curve_fit
from collections import Counter

#region growth

class Pedestal:
    """
    Look at dark images,
    See how the edge effects are important
    gaussian fit to pedestal --> cutoff = where it hits x axis
    Find the intensity counts lost and the uncertainty of lost photons here
    """
    def __init__(self, imageMatrix, title_matrix, bins, pedstalCutoffOffset=None):
        self.imageMatrix = imageMatrix
        self.title_matrix = title_matrix
        self.bins = bins
        self.pedstalCutoff = pedstalCutoffOffset

    def printHistogram(self):
        if self.pedstalCutoff is not None:
            hist_values, bin_edges = np.histogram(self.imageMatrix.flatten()[:self.pedstalCutoff], bins=self.bins)
            # Finding the peak x,y values

            maxAmpIndex = np.argmax(hist_values)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            pedestal_values = hist_values[:maxAmpIndex + self.pedstalCutoff]
            pedestal_centers = bin_centers[:maxAmpIndex + self.pedstalCutoff]
            bar_widths = np.diff(bin_edges[:maxAmpIndex + self.pedstalCutoff + 1])

            plt.figure(figsize=(10,5))

            # Raw Hist
            plt.subplot(1,2,1)
            plt.hist(self.imageMatrix.flatten(), self.bins)
            plt.yscale('log')
            plt.title(f"{self.title_matrix}")

            # Hist with cutoff
            plt.subplot(1, 2, 2)
            plt.bar(x=pedestal_centers,height=pedestal_values,width=bar_widths,align='center')
            plt.show()
        else:
            plt.hist(self.imageMatrix.flatten(), self.bins)
            plt.yscale('log')
            plt.title(f"{self.title_matrix}")
            plt.show()

    def findGaussian(self,plotGaussOverHist=False,logarithmic=False,poissonPlot=False):
        print("-"*30)
        print(f"Finding the gaussian for the pedestal for {self.title_matrix} with bins={self.bins}")
        # Obtain Histogram Data, the function gives the edges of the bin
        hist_values, bin_edges = np.histogram(self.imageMatrix.flatten(), bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # print(hist_values)

        # Finding the peak x,y values
        # This assumes the bins are not too thin such that we get distorted results
        maxAmp = max(hist_values)
        maxAmpIndex = np.argmax(hist_values)
        maxAmpBin = bin_centers[maxAmpIndex]

        print(f"The peak has index {maxAmpIndex} and value {maxAmp}")

        binRight_edge = bin_centers[3*maxAmpIndex]

        # Limiting the data to the peak

        if self.pedstalCutoff is not None:
            pedestal_values = hist_values[:maxAmpIndex + self.pedstalCutoff]
            pedestal_centers = bin_centers[:maxAmpIndex + self.pedstalCutoff]

        def gaussianFunction(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        params_initial = [
            maxAmp,     # a
            maxAmpBin,     # x0
            5     # sigma
        ]

        params, covariance = curve_fit(gaussianFunction, pedestal_centers, pedestal_values, p0=params_initial)

        aOptimised, x0Optimised, sigmaOptimised = params
        unc = np.sqrt(np.diag(covariance))
        aOptimised_unc, x0Optimised_unc, sigmaOptimised_unc = unc

        print(f"The Gaussian fit has:")
        print(f"amplitude {aOptimised} +- {aOptimised_unc}")
        print(f"mean {x0Optimised} +- {x0Optimised_unc}")
        print(f"sigma {sigmaOptimised} +- {sigmaOptimised_unc}")

        x_1sigma = x0Optimised+sigmaOptimised
        x_1sigma_unc = x0Optimised_unc+sigmaOptimised_unc
        x_2sigma = x0Optimised+2*sigmaOptimised
        x_2sigma_unc = x0Optimised_unc+2*sigmaOptimised_unc
        x_3sigma = x0Optimised+3*sigmaOptimised
        x_3sigma_unc = x0Optimised_unc+3*sigmaOptimised_unc
        print(f"ADU value of x0 + 1 standard deviation = {x_1sigma} +- {x_1sigma_unc} (~15.87% noise or {15.8/100 * 2048*2048} pixels)")
        print(f"ADU value of x0 + 2 standard deviations = {x_2sigma} +- {x_2sigma_unc} (~2.28% noise or {2.28/100 * 2048*2048} pixels)")
        print(f"ADU value of x0 + 3 standard deviations = {x_3sigma} +- {x_3sigma_unc} (~0.135% noise or {0.135/100 * 2048*2048} pixels)")


        if plotGaussOverHist:
            xvals_fitted = np.linspace(0, binRight_edge, 200)
            yvals_fitted = gaussianFunction(xvals_fitted, *params)

            plt.bar(bin_centers, hist_values, width=np.diff(bin_edges), alpha=0.6, label="Histogram")
            plt.plot(xvals_fitted, yvals_fitted, 'r-', label="Gaussian Fit")
            if logarithmic:
                plt.yscale("log")  # Match the original scale if needed
            if poissonPlot:
                xvals = np.linspace(0, 300)

            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.ylim(1)
            plt.legend()
            plt.title(self.title_matrix + f" bins={self.bins}, fitted until peak + {self.pedstalCutoff} indices")
            plt.show()

        gaussFit_dict = {
            "mean": [x0Optimised, x0Optimised_unc],
            "amplitude": [aOptimised, aOptimised_unc],
            "sigma": [sigmaOptimised, sigmaOptimised_unc],
            "x_1sigma": [x_1sigma, x_1sigma_unc],
            "x_2sigma": [x_2sigma, x_2sigma_unc],
            "x_3sigma": [x_3sigma, x_3sigma_unc]
        }

        print("gaussFit_dict keys: ", gaussFit_dict.keys())
        return gaussFit_dict

    def findGaussianSubRegion(self):
        pass

def kernelDict():
    """
    In this function sp will refer to Single pixel, dp,tp,qp etc
    :return:
    """

    # Single Pixel hits:
    sp_isolated_kernel = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])

    # TODO Consider how I want to separate diagonal terms
    sp_diagonal_kernel1 = np.array([[0, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 0]])

    sp_diagonal_kernel1 = np.array([[0, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 0]])

    # Double Pixel Hits:
    dp_isolated_kernel1 = np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0]])
    dp_isolated_kernel2 = np.rot90(dp_isolated_kernel1)
    # print(dp_isolated_kernel2)

    tp_isolated_kernel1 = np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0]])
    tp_isolated_kernel2 = np.rot90(tp_isolated_kernel1)
    # print(tp_isolated_kernel2)
    tp_isolated_kernel3 = np.rot90(tp_isolated_kernel2)
    # print(tp_isolated_kernel3)
    tp_isolated_kernel4 = np.rot90(tp_isolated_kernel3)

    qp_isolated_kernel1 = np.array([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]])

    kernel_dict = {
        "single_pixel": [sp_isolated_kernel],
        "double_pixel": [dp_isolated_kernel1, dp_isolated_kernel2],
        "triple_pixel": [tp_isolated_kernel1, tp_isolated_kernel2, tp_isolated_kernel3, tp_isolated_kernel4],
        "quadruple_pixel": [qp_isolated_kernel1]
    }

    return kernel_dict


kdic = kernelDict()

class PhotonCounting:
    def __init__(self, indexOfInterest,sp_adu_thr=180,dp_adu_thr=240,removeRows0To_=0):
        print("class PhotonCounting initiated with:")
        print("Image of Interest: ", indexOfInterest)
        print("Single Photon ADU Threshold: ", sp_adu_thr)
        print("Double Photon ADU Threshold: ", dp_adu_thr)
        self.imMatRAW = loadData()[indexOfInterest]

        if removeRows0To_ > 0:
            self.imMatRAW = self.imMatRAW[removeRows0To_:, :]

        self.sp_adu_thr = sp_adu_thr
        self.dp_adu_thr = dp_adu_thr

        def findGaussPedestal_awayFromLines():
            # 500 chosen as an approximate value that it gets fuzzy after
            iIndexStart = 500
            iIndexEnd = 1750
            jIndexStart = 50
            jIndexEnd = 1150
            matrixOfInterest = self.imMatRAW[iIndexStart:iIndexEnd, jIndexStart:jIndexEnd]
            titleH = f"Image {indexOfInterest} Gaussian Fit for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "
            ped8_indexed = Pedestal(matrixOfInterest, titleH, bins=200, pedstalCutoffOffset=15, )
            return ped8_indexed.findGaussian(logarithmic=True)

        gaussFitDict = findGaussPedestal_awayFromLines()
        self.meanPedestal = gaussFitDict["mean"][0] + gaussFitDict["mean"][1]
        self.sigmaPedestal = gaussFitDict["sigma"][0] + gaussFitDict["sigma"][1]

        def removeMeanPedestal():
            mat_minusMean = self.imMatRAW.astype(np.int16) - self.meanPedestal
            mat_minusMean[mat_minusMean < 0] = 0

            return mat_minusMean

        self.imMatMeanRemoved = removeMeanPedestal()
        self.imMat1Sigma = np.where(self.imMatMeanRemoved > self.sigmaPedestal, self.imMatMeanRemoved, 0)
        self.imMat2Sigma = np.where(self.imMatMeanRemoved > 2 * self.sigmaPedestal, self.imMatMeanRemoved, 0)
        self.imMat3Sigma = np.where(self.imMatMeanRemoved > 3 * self.sigmaPedestal, self.imMatMeanRemoved, 0)

        # Calling this self.imMat as well due to old version
        self.imMat = self.imMat3Sigma



    def checkKernels(self, kernelDictionary=None, printImages=False):

        if kernelDictionary is None:
            kernelDictionary = kernelDict()

        image_binary = np.where(self.imMat > 0, 1, 0)
        rowNum, colNum = self.imMat.shape

        matrixDictionary = {}

        for key in kernelDictionary.keys():
            print("-" * 30)
            print(f"Performing Convolution for {key} kernels:")
            kernels = kernelDictionary[key]

            matConvolvedTotal = np.zeros(self.imMat.shape)
            matchCount = 0

            for kernel in kernels:
                outputMat = np.zeros(self.imMat.shape)
                k_rows, k_cols = kernel.shape


                # Convolved the image
                for i in range(rowNum - k_rows + 1):
                    for j in range(colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = image_binary[i:i + k_rows, j:j + k_cols]

                        # Check for an exact match of the kernel shape
                        if np.array_equal(convolvedArea, kernel):
                            # If a match is found, copy the original intensities into the output matrix
                            outputMat[i:i + k_rows, j:j + k_cols] = self.imMat[i:i + k_rows, j:j + k_cols]
                            matchCount += 1

                matConvolvedTotal += outputMat

            print(f"{matchCount} matches for {key} kernels")
            matrixDictionary[key] = matConvolvedTotal

            if printImages:
                Investigate().plotMatClear(matConvolvedTotal,f"{key} kernel convolution")

        return matrixDictionary


    def checKernelType(self, kernelType, returnMatrix=False):

        print(f"The kernel type is {kernelType}")

        rowNum, colNum = self.imMat.shape
        image_binary = np.where(self.imMat > 0, 1, 0)

        list_countij = []

        if kernelType == "single_pixel":
            kernels = kernelDict()["single_pixel"]
            for kernel in kernels:
                # outputMat = np.zeros(self.imMat.shape)
                k_rows, k_cols = kernel.shape

                # Convolve the image
                for i in range(rowNum - k_rows + 1):
                    for j in range(colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = image_binary[i:i + k_rows, j:j + k_cols]

                        # Check for an exact match of the kernel shape
                        if np.array_equal(convolvedArea, kernel):
                            # If a match is found, copy the original intensities into the output matrix
                            # outputMat[i:i + k_rows, j:j + k_cols] = self.imMat[i:i + k_rows, j:j + k_cols]
                            singlePixelVal = self.imMat[i+1, j+1]
                            # print(singlePixelVal)

                            # TODO improve how this is counted. It is currently wrong. SPSP seems to be ~200

                            if singlePixelVal <= self.sp_adu_thr:
                                list_countij.append([1,i+1,j+1])
                            elif self.sp_adu_thr < singlePixelVal <= self.dp_adu_thr:
                                list_countij.append([2, i + 1, j + 1])
                            else:
                                list_countij.append([3, i + 1, j + 1])
            # print(list_countij)
            return list_countij

        if kernelType == "double_pixel":
            kernels = kernelDict()["double_pixel"]
            outputMat = np.zeros(self.imMat.shape)
            for kernel in kernels:
                k_rows, k_cols = kernel.shape
                # Convolve the image
                for i in range(rowNum - k_rows + 1):
                    for j in range(colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = image_binary[i:i + k_rows, j:j + k_cols]
                        # Check for an exact match of the kernel shape
                        if np.array_equal(convolvedArea, kernel):
                            # If a match is found, copy the original intensities into the output matrix
                            if returnMatrix:
                                outputMat[i:i + k_rows, j:j + k_cols] = self.imMat[i:i + k_rows, j:j + k_cols]

                            if k_rows == 3:
                                # Horizontal case
                                # Value on the left
                                AVal = self.imMat[i+1, j+1]
                                # Value on the Right
                                BVal = self.imMat[i+1, j+2]

                                Aindexi = i+1
                                Aindexj = j+1
                                Bindexi = i+1
                                Bindexj = j+2
                            elif k_rows == 4:
                                # Vertical Case
                                AVal = self.imMat[i+1, j+1]
                                BVal = self.imMat[i+2, j+1]
                                Aindexi = i+1
                                Aindexj = j+1
                                Bindexi = i+2
                                Bindexj = j+1
                            else:
                                print("The kernel Matrix did not have 3 or 4 rows")
                                print("k_rows = ", k_rows, " k_cols = ", k_cols)
                                print("k_rows type", type(k_rows))
                                print(kernel)
                                raise ValueError

                            totVal = AVal + BVal
                            if totVal < self.sp_adu_thr:
                                if AVal > BVal:
                                    list_countij.append([1, Aindexi, Aindexj])
                                elif BVal > AVal:
                                    list_countij.append([1, Bindexi, Bindexj])

                            if totVal > self.sp_adu_thr:
                                if AVal > BVal:
                                    list_countij.append([2, Aindexi, Aindexj])
                                elif BVal > AVal:
                                    list_countij.append([2, Bindexi, Bindexj])

            if returnMatrix:
                return outputMat
            # print(list_countij)
            return list_countij

        if kernelType == "triple_pixel":
            # Intialise triple pixel kernels
            kernels = kernelDict()["triple_pixel"]
            # Initialise a matrix where we could store only 3 pixel points if return Matrx = True
            outputMat = np.zeros(self.imMat.shape)
            # Initialise count for lost points
            countLost = 0

            for kernel in kernels:
                k_rows, k_cols = kernel.shape
                # Convolve the image
                for i in range(rowNum - k_rows + 1):
                    for j in range(colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = image_binary[i:i + k_rows, j:j + k_cols]
                        # Check for an exact match of the kernel shape
                        if np.array_equal(convolvedArea, kernel):
                            # If a match is found, copy the original intensities into the output matrix
                            if returnMatrix:
                                outputMat[i:i + k_rows, j:j + k_cols] = self.imMat[i:i + k_rows, j:j + k_cols]

                            # TODO Consider a certain threshold for value at the center where we want to consider lower thresholded image

                            if np.array_equal(kernel,np.array([[0, 0, 0, 0],[0, 1, 1, 0],[0, 0, 1, 0],[0, 0, 0, 0]])):
                                # This is the case there the double adjacent piece is in the top right
                                doubleAdjVal = self.imMat[i+1, j+2]
                                doubleAdjIndex_ij = [i+1,j+2]
                                aVal = self.imMat[i+1, j+1]
                                bVal = self.imMat[i+2, j+2]
                            elif np.array_equal(kernel,np.array([[0, 0, 0, 0],[0, 0, 1, 0],[0, 1, 1, 0],[0, 0, 0, 0]])):
                                # This is the case there the double adjacent piece is in the bottom right
                                doubleAdjVal = self.imMat[i+2, j+2]
                                doubleAdjIndex_ij = [i+2,j+2]
                                aVal = self.imMat[i+1, j+2]
                                bVal = self.imMat[i+2, j+1]
                            elif np.array_equal(kernel,np.array([[0, 0, 0, 0],[0, 1, 0, 0],[0, 1, 1, 0],[0, 0, 0, 0]])):
                                # This is the case there the double adjacent piece is in the bottom left
                                doubleAdjVal = self.imMat[i+2, j+1]
                                doubleAdjIndex_ij = [i+2,j+1]
                                aVal = self.imMat[i+2, j+2]
                                bVal = self.imMat[i+1, j+1]
                            else:
                                # This is the case there the double adjacent piece is in the top left
                                doubleAdjVal = self.imMat[i + 1, j + 1]
                                doubleAdjIndex_ij = [i + 1, j + 1]
                                aVal = self.imMat[i + 1, j + 2]
                                bVal = self.imMat[i + 2, j + 1]

                            totVal = doubleAdjVal + aVal + bVal

                            if not (doubleAdjVal > aVal and doubleAdjVal > bVal):
                                if totVal < self.sp_adu_thr:
                                    list_countij.append([1,doubleAdjIndex_ij[0],doubleAdjIndex_ij[1]])
                                elif totVal > self.sp_adu_thr:
                                    list_countij.append([2, doubleAdjIndex_ij[0], doubleAdjIndex_ij[1]])
                            else:
                                countLost +=1


            print(f"The number of points where the double adjacent point wasn't the greatest was {countLost}")

            if returnMatrix:
                return outputMat
            # print(list_countij)
            return list_countij


    def CheckKernelList(self, kernelList, title_kernelList="", printImages=False):
        image_binary = np.where(self.imMat > 0, 1, 0)
        rowNum, colNum = self.imMat.shape

        matrixDictionary = {}

        matConvolvedTotal = np.zeros(self.imMat.shape)
        matchCount = 0

        for kernel in kernelList:
            outputMat = np.zeros(self.imMat.shape)
            k_rows, k_cols = kernel.shape

            # Convolved the image
            for i in range(rowNum - k_rows + 1):
                for j in range(colNum - k_cols + 1):
                    # Consider areas of the same size as the kernel:
                    convolvedArea = image_binary[i:i + k_rows, j:j + k_cols]

                    # Check for an exact match of the kernel shape
                    if np.array_equal(convolvedArea, kernel):
                        # If a match is found, copy the original intensities into the output matrix
                        outputMat[i:i + k_rows, j:j + k_cols] = self.imMat[i:i + k_rows, j:j + k_cols]
                        matchCount += 1

            matConvolvedTotal += outputMat

        print(f"{matchCount} matches")

        if printImages:

            fig, ax = plt.subplots(figsize=(8, 8))
            # Get indices of nonzero elements
            y, x = np.nonzero(matConvolvedTotal)
            values = matConvolvedTotal[y, x]  # Get intensity values for color mapping

            scatter = ax.scatter(x, y, c=values, cmap='plasma', s=5, edgecolors='white', linewidth=0.2)
            plt.colorbar(scatter, ax=ax, label="Intensity")
            # Invert y-axis to match image orientation
            ax.set_ylim(ax.get_ylim()[::-1])

            # Set xticks and yticks to correspond to matrix indices
            ax.set_xticks(np.linspace(0, self.imMat.shape[1], 10))  # Set 10 evenly spaced ticks
            ax.set_yticks(np.linspace(0, self.imMat.shape[0], 10))

            ax.set_xticklabels([int(i) for i in np.linspace(0, self.imMat.shape[1], 10)])  # Ensure integer labels
            ax.set_yticklabels([int(i) for i in np.linspace(0, self.imMat.shape[0], 10)])

            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")
            ax.set_title(title_kernelList)
            ax.set_aspect('equal')  # Ensure square pixels
            plt.show()

        return matConvolvedTotal

    def singlePhotonSinglePixelHits(self):
        """
        Takes thresholded image matrix and finds all isolated points with no non zero neighbors
        :return: A list of indices [i,j] which corresponds to [y,x] from the top left corner
        """

        rows = self.imMat.shape[0]
        cols = self.imMat.shape[1]

        # initialise list to store isolated points
        isolated_points = []

        # Iterate over each pixel in the image (excluding borders)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Get the intensity of the current pixel
                current_intensity = self.imMat[i, j]

                # If the current intensity is non-zero, check its neighbors
                if current_intensity != 0:
                    # Get the 8 neighbors around the pixel
                    neighbors = [
                        self.imMat[i - 1, j - 1], self.imMat[i - 1, j], self.imMat[i - 1, j + 1],
                        self.imMat[i, j - 1], self.imMat[i, j + 1],
                        self.imMat[i + 1, j - 1], self.imMat[i + 1, j], self.imMat[i + 1, j + 1]
                    ]

                    # If all neighbors are 0, the point is isolated
                    if all(neighbor == 0 for neighbor in neighbors):
                        isolated_points.append([i, j])

        print(f"Single Photon Single Pixel Hits: {len(isolated_points)}")
        return isolated_points

    class IterativeRemoval:
        "The idea of this code is to try and intelligently search from top down to remove elements"
        "For example elements of order 180 on the ADU are likely to be either a single pixel single photon hit"
        "or in the case that there is an adjacent photon of similar order than a double photon hit"
        "KEY point for this idea is we remove these from the image. This hence demands some human input into what"
        "the ADU value of that regime would look like"


# TODO consider more complex non isolated setupsh
# TODO Try train a model to learn how to find what a photon is



if __name__ == "__main__":


    def compareRawToSPSP():
        spc = PhotonCounting(high_intensity_points)
        matDict = spc.checkKernels(printImages=False)
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1), plt.imshow(high_intensity_points, cmap='hot'), plt.title(f"Image 8 Thresholded")
        # plt.subplot(1, 2, 2), plt.imshow(matDict["double_pixel"], cmap='hot'), plt.title(f"Double Pixel Hits")
        # plt.show()


    # compareRawToSPSP()


    def testMethods(indexOfInterest=8):

        print(f"Finding the counts for image {indexOfInterest}")
        spc = PhotonCounting(indexOfInterest)
        print("-" * 30)

        listCount1 = spc.checKernelType("single_pixel")
        count_occurrences1 = Counter(countij[0] for countij in listCount1)
        print("Count Occurences for single Pixel Hits")
        print(count_occurrences1)

        print("-"*30)

        listCount2 = spc.checKernelType("double_pixel")
        count_occurrences2 = Counter(countij[0] for countij in listCount2)
        print("Count Occurences for double Pixel Hits")
        print(count_occurrences2)

        print("-" * 30)

        listCount3 = spc.checKernelType("triple_pixel")
        count_occurrences3 = Counter(countij[0] for countij in listCount3)
        print("Count Occurences for triple Pixel Hits")
        print(count_occurrences3)


    testMethods()

    def testFindPedestal(indexOfInterest):
        imageMatrix = imData[indexOfInterest]
        iIndexStart = 500
        iIndexEnd = 1750
        jIndexStart = 50
        jIndexEnd = 1150
        matrixOfInterest = imageMatrix[iIndexStart:iIndexEnd, jIndexStart:jIndexEnd]
        titleH = f"Image {indexOfInterest} Gaussian Fit for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "

        ped8 = Pedestal(imageMatrix, "Image 8", bins=200, pedstalCutoffOffset=15, )
        ped8.findGaussian(logarithmic=True)

        ped8_indexed = Pedestal(matrixOfInterest,titleH,bins=200,pedstalCutoffOffset=15, )
        ped8_indexed.findGaussian(logarithmic=True)

    # testFindPedestal(8)

    def testInitSPC(logarithmic=True):
        ped = PhotonCounting(8,)
        imMat = ped.imMat

        vals = imMat.flatten()
        vals = vals[vals > 0]
        plt.hist(vals, 200)
        if logarithmic:
            plt.yscale('log')
        plt.title("test matrix histogram for mean subtracted spectrum")
        plt.show()

    # testInitSPC()

    pass
