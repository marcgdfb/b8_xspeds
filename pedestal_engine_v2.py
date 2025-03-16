from imagePreProcessing import *
from scipy.optimize import curve_fit
from probability_tools import *


class Pedestal:
    def __init__(self, imageMatrix, title_matrix, bins=300, pedestalOffset_adu=None):
        self.imageMatrix = imageMatrix
        self.title_matrix = title_matrix
        self.bins = bins

        if pedestalOffset_adu is None:
            self.pedestal_offset_adu = 25
        else:
            self.pedestal_offset_adu = pedestalOffset_adu

    def findGaussian(self, plotGaussOverHist=False, logarithmic=False, diagnostics=False):
        """
        Takes the image matrix and given the bins calculates the gaussian fit. The pedestal offset is the value of adu
        after the maximum taken into consideration.

        :return: A dictionary with keys 'mean', 'amplitude', 'sigma', 'x_1sigma', 'x_2sigma', 'x_3sigma'
        Each value is a list where the 0th value is its expected value and the 1st value is its uncertainty / standard deviation
        """
        print("-" * 30)
        print(f"Finding the gaussian for the pedestal for {self.title_matrix} with bins={self.bins} and adu offset {self.pedestal_offset_adu}")
        # Obtain Histogram Data, the function gives the edges of the bin
        hist_values, bin_edges = np.histogram(self.imageMatrix.flatten(), bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Finding the peak x,y values
        # This assumes the bins are not too thin such that we get distorted results
        maxAmp = max(hist_values)
        maxAmpIndex = np.argmax(hist_values)
        maxAmpBin = bin_centers[maxAmpIndex]

        bin_width = bin_edges[1] - bin_edges[0]
        pedestal_offset_index = round(self.pedestal_offset_adu/bin_width) + maxAmpIndex

        # Limiting the data to the peak
        pedestal_values = hist_values[:pedestal_offset_index]
        pedestal_centers = bin_centers[:pedestal_offset_index]

        if diagnostics:
            print("max bin edge", max(bin_edges))
            print("max bin edge / bin count", max(bin_edges) / self.bins)
            print("bin_width", bin_width)
            print("pedestal_offset_index", pedestal_offset_index)
            print("pedestal_offset_adu", bin_centers[pedestal_offset_index])
            print("maxAmpIndex", maxAmpIndex)
            plt.bar(pedestal_centers, pedestal_values, width=bin_width, alpha=0.6, label="Pedestal Histogram")
            plt.show()


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

        # print(f"amplitude {aOptimised} +- {aOptimised_unc}")
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

        nrows = self.imageMatrix.shape[0]
        ncols = self.imageMatrix.shape[1]
        npixels = nrows * ncols

        above1,above2,above3 = findExpectedCountsAbove123sigma(gaussFit_dict,npixels)

        print(f"\nThe ADU value associated with x0 + N sigma and the number of counts found above that is:")
        print(f"1: {gaussFit_dict['x_1sigma'][0]:.2f} +- {gaussFit_dict['x_1sigma'][1]:.2f} with {above1} counts")
        print(f"2: {gaussFit_dict['x_2sigma'][0]:.2f} +- {gaussFit_dict['x_2sigma'][1]:.2f} with {above2} counts")
        print(f"3: {gaussFit_dict['x_3sigma'][0]:.2f} +- {gaussFit_dict['x_3sigma'][1]:.2f} with {above3} counts")

        if plotGaussOverHist:
            binEdgeFitted = bin_edges[pedestal_offset_index]
            binRight_edge = bin_centers[3 * maxAmpIndex]
            x_vals_fitted = np.linspace(0,binEdgeFitted, 200)
            yvals_fitted = gaussianFunction(x_vals_fitted, *params)

            xvals_extrapolated = np.linspace(binEdgeFitted, binRight_edge, 200)
            yvals_extrapolated = gaussianFunction(xvals_extrapolated, *params)

            plt.bar(bin_centers, hist_values, width=np.diff(bin_edges), alpha=0.6, label="Histogram")

            plt.plot(x_vals_fitted, yvals_fitted, 'r-', label="Gaussian Fit")
            plt.plot(xvals_extrapolated, yvals_extrapolated, 'g-', label="Gaussian Fit Extrapolated")
            if logarithmic:
                plt.yscale("log")  # Match the original scale if needed

            plt.xlabel("ADU value")
            plt.ylabel("Frequency")
            plt.ylim(0.5)
            plt.legend()
            plt.title(self.title_matrix + f"\nbins={self.bins}, fitted until peak + {self.pedestal_offset_adu} ADU")
            plt.show()

        # print("gaussFit_dict keys: ", gaussFit_dict.keys())
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


    def printHistogram(self):
        if self.pedestal_offset_adu is not None:
            hist_values, bin_edges = np.histogram(self.imageMatrix.flatten()[:self.pedestal_offset_adu], bins=self.bins)
            # Finding the peak x,y values

            maxAmpIndex = np.argmax(hist_values)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            pedestal_values = hist_values[:maxAmpIndex + self.pedestal_offset_adu]
            pedestal_centers = bin_centers[:maxAmpIndex + self.pedestal_offset_adu]
            bar_widths = np.diff(bin_edges[:maxAmpIndex + self.pedestal_offset_adu + 1])

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


def pedestal_mean_sigma_awayFromLines(imMatrix, indexOfInterest):
    # The i starting index is to avoid the edge effects at the top which is prominent
    iIndexStart, iIndexEnd, jIndexStart, jIndexEnd = 500, 1750, 50, 1150
    matrixOfInterest = imMatrix[iIndexStart:iIndexEnd, jIndexStart:jIndexEnd]
    titleH = f"Image {indexOfInterest} Gaussian Fit for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "
    ped8_indexed = Pedestal(matrixOfInterest, titleH, bins=300, pedestalOffset_adu=20, )
    gaussFitDict_ = ped8_indexed.findGaussian(logarithmic=True)

    meanPedestal = gaussFitDict_["mean"][0]  # + gaussFitDict["mean"][1]
    sigmaPedestal = gaussFitDict_["sigma"][0]  # + gaussFitDict["sigma"][1]

    return meanPedestal, sigmaPedestal


def mat_thr_aboveNsigma(index_of_interest, how_many_sigma, ):
    image_mat = loadData()[index_of_interest]
    # ----------pedestal mean and sigma----------
    ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image_mat, index_of_interest)
    thr = ped_mean + how_many_sigma * ped_sigma

    image_mat = np.where(image_mat > thr, image_mat, 0)

    return image_mat, thr

def mat_minusMean_thr_aboveNsigma(index_of_interest, how_many_sigma, ):
    image_mat = loadData()[index_of_interest]
    # ----------pedestal mean and sigma----------
    ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image_mat, index_of_interest)
    thr = how_many_sigma * ped_sigma

    mat_minusMean = image_mat.astype(np.int16) - ped_mean
    mat_minusMean[mat_minusMean < 0] = 0

    mat_minusMean = np.where(mat_minusMean > thr, mat_minusMean, 0)

    return mat_minusMean, thr

def create_sum_all_images():
    matzeros = np.zeros((2048,2048))
    for index in range(len(loadData())):
        # image_mat, __ = mat_thr_aboveNsigma(index, 2)

        image_mat = loadData()[index]
        matzeros += image_mat

    plt.imshow(matzeros, cmap='hot')
    plt.title("Matrix Sum of All Raw Images")
    plt.ylabel("Image j Index")
    plt.xlabel("Image i Index")
    plt.show()


if __name__ == "__main__":

    def check_spcUsage(indexOfInterest):
        imMatRaw = loadData()[indexOfInterest]
        iIndexStart = 500
        iIndexEnd = 1750
        jIndexStart = 50
        jIndexEnd = 1150
        matrixOfInterest = imMatRaw[iIndexStart:iIndexEnd, jIndexStart:jIndexEnd]
        titleH = f"Image {indexOfInterest} Gaussian Fit for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "
        ped8_indexed = Pedestal(matrixOfInterest, titleH, bins=300, pedestalOffset_adu=25, )
        ped8_indexed.findGaussian(plotGaussOverHist=True, logarithmic=True,diagnostics=False)

    # check_spcUsage(8)

    # image_mat11, thr2sigma11 = mat_minusMean_thr_aboveNsigma(11,2)
    # plt.imshow(image_mat11,cmap='hot')
    # plt.show()

    # create_sum_all_images()

    pass

