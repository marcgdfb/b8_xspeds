from imagePreProcessing import *
from scipy.optimize import curve_fit
from collections import Counter
from probability_tools import *
from tools import *


# TODO: kernel not to give up terms with a diagonal adjacent

class Pedestal:
    """
    Look at dark images,
    See how the edge effects are important
    gaussian fit to pedestal --> cutoff = where it hits x-axis
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

            plt.figure(figsize=(10, 5))

            # Raw Hist
            plt.subplot(1, 2, 1)
            plt.hist(self.imageMatrix.flatten(), self.bins)
            plt.yscale('log')
            plt.title(f"{self.title_matrix}")

            # Hist with cutoff
            plt.subplot(1, 2, 2)
            plt.bar(x=pedestal_centers, height=pedestal_values, width=bar_widths, align='center')
            plt.show()
        else:
            plt.hist(self.imageMatrix.flatten(), self.bins)
            plt.yscale('log')
            plt.title(f"{self.title_matrix}")
            plt.show()

    def findGaussian(self, plotGaussOverHist=False, logarithmic=False, poissonPlot=False):
        print("-" * 30)
        print(f"Finding the gaussian for the pedestal for {self.title_matrix} with bins={self.bins} and offset {self.pedstalCutoff}")
        # Obtain Histogram Data, the function gives the edges of the bin
        hist_values, bin_edges = np.histogram(self.imageMatrix.flatten(), bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # print(hist_values)

        # Finding the peak x,y values
        # This assumes the bins are not too thin such that we get distorted results
        maxAmp = max(hist_values)
        maxAmpIndex = np.argmax(hist_values)
        maxAmpBin = bin_centers[maxAmpIndex]

        # print(f"The peak has index {maxAmpIndex} and value {maxAmp}")

        binRight_edge = bin_centers[3 * maxAmpIndex]

        # Limiting the data to the peak

        if self.pedstalCutoff is not None:
            pedestal_values = hist_values[:maxAmpIndex + self.pedstalCutoff]
            pedestal_centers = bin_centers[:maxAmpIndex + self.pedstalCutoff]

        def gaussianFunction(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        params_initial = [
            maxAmp,  # a
            maxAmpBin,  # x0
            5  # sigma
        ]

        params, covariance = curve_fit(gaussianFunction, pedestal_centers, pedestal_values, p0=params_initial)

        aOptimised, x0Optimised, sigmaOptimised = params
        unc = np.sqrt(np.diag(covariance))
        aOptimised_unc, x0Optimised_unc, sigmaOptimised_unc = unc

        print(f"The Gaussian fit has:")
        print(f"amplitude {aOptimised} +- {aOptimised_unc}")
        print(f"mean {x0Optimised} +- {x0Optimised_unc}")
        print(f"sigma {sigmaOptimised} +- {sigmaOptimised_unc}")

        gaussFit_dict = {
            "mean": [x0Optimised, x0Optimised_unc],
            "amplitude": [aOptimised, aOptimised_unc],
            "sigma": [sigmaOptimised, sigmaOptimised_unc],
            "x_1sigma": [x0Optimised + sigmaOptimised, x0Optimised_unc + sigmaOptimised_unc],
            "x_2sigma": [x0Optimised + 2 * sigmaOptimised, x0Optimised_unc + 2 * sigmaOptimised_unc],
            "x_3sigma": [x0Optimised + 3 * sigmaOptimised, x0Optimised_unc + 3 * sigmaOptimised_unc]
        }

        above1,above2,above3 = findExpectedCountsAbove123sigma(gaussFit_dict)

        nrows,ncols = self.imageMatrix.shape
        npixels = nrows*ncols

        print(f"ADU value of x0 + 1 standard deviation = {gaussFit_dict["x_1sigma"][0]} +- {gaussFit_dict["x_1sigma"][1]}")
        print(f"We expect {above1} counts above 1 sigma i.e. {above1/npixels} of pixels")
        print(f"ADU value of x0 + 2 standard deviations = {gaussFit_dict["x_2sigma"][0]} +- {gaussFit_dict["x_2sigma"][1]}")
        print(f"We expect {above2} counts above 2 sigma i.e. {above2/npixels} of pixels")
        print(f"ADU value of x0 + 3 standard deviations = {gaussFit_dict["x_3sigma"][0]} +- {gaussFit_dict["x_3sigma"][1]}")
        print(f"We expect {above3} counts above 3 sigma i.e. {above3/npixels} of pixels")

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


        print("gaussFit_dict keys: ", gaussFit_dict.keys())
        return gaussFit_dict


    def compareSimulatedNoise(self,threshold_num_sigma=2):

        gaussFit_dict = self.findGaussian()

        x0 = gaussFit_dict["mean"][0]
        sigma = gaussFit_dict["sigma"][0]

        matSimulated = generateImageGauss(x0,sigma,threshold_num_sigma=threshold_num_sigma)

        # imMat_Thr = np.where((self.imageMatrix > x0+threshold_num_sigma*sigma) & (self.imageMatrix < x0+(threshold_num_sigma+1)*sigma), self.imageMatrix, 0)
        imMat_Thr = np.where(self.imageMatrix > x0+threshold_num_sigma*sigma, self.imageMatrix, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(matSimulated, cmap='hot'), plt.title(f"Simulated Noise using Pedestal above {threshold_num_sigma} sigma")
        plt.subplot(1, 2, 2), plt.imshow(imMat_Thr,cmap='hot'), plt.title(f"Image Thresholded above {threshold_num_sigma} sigma")
        plt.show()


    def compareSigma(self,sigma_count1,sigma_count2):
        gaussFit_dict = self.findGaussian()

        x0 = gaussFit_dict["mean"][0]
        sigma = gaussFit_dict["sigma"][0]

        imMat_ThrA = np.where(self.imageMatrix > x0 + sigma_count1 * sigma, self.imageMatrix, 0)
        imMat_ThrB = np.where(self.imageMatrix > x0 + sigma_count2 * sigma, self.imageMatrix, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(imMat_ThrA, cmap='hot'), plt.title(f"Simulated Noise using Pedestal above {sigma_count1} sigma")
        plt.subplot(1, 2, 2), plt.imshow(imMat_ThrB, cmap='hot'), plt.title(f"Image Thresholded above {sigma_count2} sigma")
        plt.show()


def kernelDict():
    """
    In this function sp will refer to Single pixel, dp,tp,qp etc.
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
    def __init__(self, indexOfInterest, sp_adu_thr=180, dp_adu_thr=240, no_photon_adu_thr=50,
                 removeRows0To_=0, howManySigma_thr=3):

        def printVar():
            print("-" * 30)
            print("class PhotonCounting initiated with:")
            print("Image of Interest: ", indexOfInterest)
            print("No Photon ADU Threshold: ", no_photon_adu_thr)
            print("Single Photon ADU Threshold: ", sp_adu_thr)
            print("Double Photon ADU Threshold: ", dp_adu_thr)
            print("Remove Rows 0 to: ", removeRows0To_)
            print(f"imMat has the mean removed and is thresholded above {howManySigma_thr} sigma ")

        printVar()

        self.imMatRAW = loadData()[indexOfInterest]

        if removeRows0To_ > 0:
            self.imMatRAW = self.imMatRAW[removeRows0To_:, :]

        self.no_p_adu_thr = no_photon_adu_thr
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
            print("Subtracting the mean of the pedestal from the image")
            mat_minusMean = self.imMatRAW.astype(np.int16) - self.meanPedestal
            mat_minusMean[mat_minusMean < 0] = 0

            return mat_minusMean

        self.imMatMeanRemoved = removeMeanPedestal()
        self.imMat1Sigma = np.where(self.imMatMeanRemoved > self.sigmaPedestal, self.imMatMeanRemoved, 0)
        self.imMat2Sigma = np.where(self.imMatMeanRemoved > 2 * self.sigmaPedestal, self.imMatMeanRemoved, 0)
        self.imMat3Sigma = np.where(self.imMatMeanRemoved > 3 * self.sigmaPedestal, self.imMatMeanRemoved, 0)

        # Calling this self.imMat as well due to old version
        self.howManySigma = howManySigma_thr
        print(f"imMat is {howManySigma_thr} sigma away from mean:")
        self.imMat = np.where(self.imMatMeanRemoved > howManySigma_thr * self.sigmaPedestal, self.imMatMeanRemoved, 0)

    def checKernelType(self, kernelType, returnMatrix=False, diagnostics=False):

        # TODO: Consider all shapes for 3, e.g.

        print(f"The kernel type is {kernelType}")

        rowNum, colNum = self.imMat.shape
        image_binary = np.where(self.imMat > 0, 1, 0)

        outputMat = np.zeros(self.imMat.shape)
        # Initialise Counts
        countReject = 0
        count_1photon = 0
        count_2photon = 0
        count_morethan2 = 0

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
                            singlePixelVal = self.imMat[i + 1, j + 1]
                            # print(singlePixelVal)

                            if singlePixelVal <= self.no_p_adu_thr:
                                countReject += 1
                                continue
                            elif self.no_p_adu_thr < singlePixelVal <= self.sp_adu_thr:
                                list_countij.append([1, i + 1, j + 1])
                            elif self.sp_adu_thr < singlePixelVal <= self.dp_adu_thr:
                                list_countij.append([2, i + 1, j + 1])
                            else:
                                list_countij.append([3, i + 1, j + 1])
            # print(list_countij)

        if kernelType == "double_pixel":
            kernels = kernelDict()["double_pixel"]
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
                                AVal = self.imMat[i + 1, j + 1]
                                # Value on the Right
                                BVal = self.imMat[i + 1, j + 2]

                                Aindexi = i + 1
                                Aindexj = j + 1
                                Bindexi = i + 1
                                Bindexj = j + 2
                            elif k_rows == 4:
                                # Vertical Case
                                AVal = self.imMat[i + 1, j + 1]
                                BVal = self.imMat[i + 2, j + 1]
                                Aindexi = i + 1
                                Aindexj = j + 1
                                Bindexi = i + 2
                                Bindexj = j + 1
                            else:
                                print("The kernel Matrix did not have 3 or 4 rows")
                                print("k_rows = ", k_rows, " k_cols = ", k_cols)
                                print("k_rows type", type(k_rows))
                                print(kernel)
                                raise ValueError

                            totVal = AVal + BVal

                            if totVal <= self.no_p_adu_thr:
                                countReject += 1
                                continue
                            elif totVal < self.sp_adu_thr:
                                if AVal > BVal:
                                    list_countij.append([1, Aindexi, Aindexj])
                                elif BVal > AVal:
                                    list_countij.append([1, Bindexi, Bindexj])
                            elif totVal < self.dp_adu_thr:
                                if AVal > BVal:
                                    list_countij.append([2, Aindexi, Aindexj])
                                elif BVal > AVal:
                                    list_countij.append([2, Bindexi, Bindexj])
                            elif totVal > self.dp_adu_thr:
                                print(totVal)
                                print(self.imMat[i:i + k_rows, j:j + k_cols])

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

                            if np.array_equal(kernel,
                                              np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])):
                                # This is the case there the double adjacent piece is in the top right
                                doubleAdjVal = self.imMat[i + 1, j + 2]
                                doubleAdjIndex_ij = [i + 1, j + 2]
                                aVal = self.imMat[i + 1, j + 1]
                                bVal = self.imMat[i + 2, j + 2]
                            elif np.array_equal(kernel,
                                                np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])):
                                # This is the case there the double adjacent piece is in the bottom right
                                doubleAdjVal = self.imMat[i + 2, j + 2]
                                doubleAdjIndex_ij = [i + 2, j + 2]
                                aVal = self.imMat[i + 1, j + 2]
                                bVal = self.imMat[i + 2, j + 1]
                            elif np.array_equal(kernel,
                                                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])):
                                # This is the case there the double adjacent piece is in the bottom left
                                doubleAdjVal = self.imMat[i + 2, j + 1]
                                doubleAdjIndex_ij = [i + 2, j + 1]
                                aVal = self.imMat[i + 2, j + 2]
                                bVal = self.imMat[i + 1, j + 1]
                            else:
                                # This is the case there the double adjacent piece is in the top left
                                doubleAdjVal = self.imMat[i + 1, j + 1]
                                doubleAdjIndex_ij = [i + 1, j + 1]
                                aVal = self.imMat[i + 1, j + 2]
                                bVal = self.imMat[i + 2, j + 1]

                            totVal = doubleAdjVal + aVal + bVal

                            if not (doubleAdjVal > aVal and doubleAdjVal > bVal):
                                if totVal < self.no_p_adu_thr:
                                    countLost += 1
                                    continue
                                elif totVal < self.sp_adu_thr:
                                    list_countij.append([1, doubleAdjIndex_ij[0], doubleAdjIndex_ij[1]])
                                elif totVal < self.dp_adu_thr:
                                    list_countij.append([2, doubleAdjIndex_ij[0], doubleAdjIndex_ij[1]])
                                else:
                                    print(totVal)
                                    print(self.imMat[i:i + k_rows, j:j + k_cols])
                            else:
                                countLost += 1
            print(f"The number of points where the double adjacent point wasn't the greatest was {countLost}")

        if kernelType == "quadruple_pixel":
            # Intialise triple pixel kernels
            kernels = kernelDict()["quadruple_pixel"]
            # Initialise a matrix where we could store only 3 pixel points if return Matrx = True
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

                            # Create dictionary with keys tl = top left, top right, bottom left etc.
                            dict_idx = {
                                "tl": [i + 1, j + 1],
                                "tr": [i + 1, j + 2],
                                "bl": [i + 2, j + 1],
                                "br": [i + 2, j + 2],
                            }

                            # initialise a dictionary of values with the same key
                            dict_vals = {}
                            totVal = 0
                            # Also initialise totVal to find the total of all 4 spots
                            for key in dict_idx.keys():
                                valOfKey = self.imMat[dict_idx[key][0], dict_idx[key][1]]
                                dict_vals[key] = valOfKey
                                totVal += valOfKey

                            if totVal < self.no_p_adu_thr:
                                countReject += 1
                                continue
                            elif totVal < self.sp_adu_thr:
                                keyOrderedList = sorted_keys_by_value(dict_vals)
                                max_key = keyOrderedList[0]

                                list_countij.append([1, dict_idx[max_key][0], dict_idx[max_key][1]])
                                count_1photon += 1
                            elif totVal < self.dp_adu_thr:
                                keyOrderedList = sorted_keys_by_value(dict_vals)

                                key_1 = keyOrderedList[0]
                                key_2 = keyOrderedList[1]

                                list_countij.append([1, dict_idx[key_1][0], dict_idx[key_1][1]])
                                list_countij.append([1, dict_idx[key_2][0], dict_idx[key_2][1]])

                                if diagnostics:
                                    print("dp_adu_thr")
                                    print(self.imMat[i:i + k_rows, j:j + k_cols])

                                count_2photon += 1
                            else:
                                if diagnostics:
                                    print("more than dp_adu_thr")
                                    print(self.imMat[i:i + k_rows, j:j + k_cols])
                                count_morethan2 += 1

        if returnMatrix:
            return outputMat

        def reportCounts():

            print("-"*30)
            print(f"The results for {kernelType} were:")
            print(f"Number of found elements rejected: {countReject}")
            print(f"Number of 1 photon elements: {count_1photon}")
            print(f"Number of 2 photon elements: {count_2photon}")
            print(f"Number of elements with more than 2 photon elements: {count_morethan2}")

        reportCounts()

        # print(list_countij)
        return list_countij

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
                Investigate().plotMatClear(matConvolvedTotal, f"{key} kernel convolution")

        return matrixDictionary

    def singlePhotonSinglePixelHits(self):
        """
        Takes thresholded image matrix and finds all isolated points with no non-zero neighbors
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
        """The idea of this code is to try and intelligently search from top down to remove elements
        For example elements of order 180 on the ADU are likely to be either a single pixel single photon hit
        or in the case that there is an adjacent photon of similar order than a double photon hit
        KEY point for this idea is we remove these from the image. This hence demands some human input into what
        the ADU value of that regime would look like"""



# TODO consider more complex non isolated setupsh
# TODO Try train a model to learn how to find what a photon is


if __name__ == "__main__":

    def testMethods(indexOfInterest=8, listMethods=None,
                    sp_adu_thr=180, dp_adu_thr=240, no_photon_adu_thr=50,
                    removeRows0To_=0, howManySigma_thr=3):

        for var, value in locals().items():
            print(f"  {var}: {value}")

        if listMethods is None:
            listMethods = ["single_pixel", "double_pixel", "triple_pixel"]

        spc = PhotonCounting(indexOfInterest, sp_adu_thr, dp_adu_thr, no_photon_adu_thr,
                             removeRows0To_, howManySigma_thr)

        for method in listMethods:
            print("-" * 30)
            listCount = spc.checKernelType(method)
            count_occurrences = Counter(countij[0] for countij in listCount)
            print(f"Count Occurences for {method}")
            print(count_occurrences)

    testMethods(8,["quadruple_pixel"],howManySigma_thr=2)


    # testMethods()

    def testFindPedestal(indexOfInterest=8):
        imageMatrix = imData[indexOfInterest]
        iIndexStart = 500
        iIndexEnd = 1750
        jIndexStart = 50
        jIndexEnd = 1150
        matrixOfInterest = imageMatrix[iIndexStart:iIndexEnd, jIndexStart:jIndexEnd]
        titleH = f"Image {indexOfInterest} Gaussian Fit for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "

        # ped8 = Pedestal(imageMatrix, "Image 8", bins=200, pedstalCutoffOffset=15, )
        # ped8.findGaussian(logarithmic=True)

        ped8_indexed = Pedestal(matrixOfInterest, titleH, bins=200, pedstalCutoffOffset=15, )
        ped8_indexed.findGaussian(plotGaussOverHist=True,logarithmic=True,)


    # testFindPedestal(9)

    def testInitSPC(logarithmic=True):

        ped = PhotonCounting(8, )
        imMat = ped.imMatMeanRemoved

        def plotLinesOfSTD():
            x_values = [ped.sigmaPedestal, 2 * ped.sigmaPedestal, 3 * ped.sigmaPedestal]
            labels = ['1 σ', '2 σ', '3 σ']  # Labels for each line
            # Plot each vertical line with a label
            for x, label in zip(x_values, labels):
                plt.axvline(x=x, color='r', linestyle='--', label=label)

            # plt.xlim(0, 8)  # Adjust x-axis limits for clarity
            # plt.ylim(0, 10)  # Adjust y-axis limits if needed
            plt.legend()

        vals = imMat.flatten()
        vals = vals[vals > 0]
        plt.hist(vals, 200)
        if logarithmic:
            plt.yscale('log')
        plt.title(f"test matrix histogram for mean subtracted spectrum \nσ = {ped.sigmaPedestal:.2f}")
        plotLinesOfSTD()
        plt.xlabel("ADU")
        plt.ylabel("Count")
        plt.show()


    # testInitSPC()

    # Pedestal(imData[8],title_matrix="",bins=200,pedstalCutoffOffset=15,).compareSimulatedNoise(2)
    # Pedestal(imData[8], title_matrix="", bins=200, pedstalCutoffOffset=15, ).compare2to3Sigma()

    pass
