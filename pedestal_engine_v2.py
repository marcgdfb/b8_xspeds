from imagePreProcessing import *
from scipy.optimize import curve_fit
from probability_tools import *
import seaborn as sns
import os

# Considering how often this is used, could save mean and sigma to help reduce run time

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

        # nrows = self.imageMatrix.shape[0]
        # ncols = self.imageMatrix.shape[1]
        # npixels = nrows * ncols

        # above1,above2,above3 = findExpectedCountsAbove123sigma(gaussFit_dict,npixels)

        # print(f"\nThe ADU value associated with x0 + N sigma and the number of counts found above that is:")
        # print(f"1: {gaussFit_dict['x_1sigma'][0]:.2f} +- {gaussFit_dict['x_1sigma'][1]:.2f} with {above1} counts")
        # print(f"2: {gaussFit_dict['x_2sigma'][0]:.2f} +- {gaussFit_dict['x_2sigma'][1]:.2f} with {above2} counts")
        # print(f"3: {gaussFit_dict['x_3sigma'][0]:.2f} +- {gaussFit_dict['x_3sigma'][1]:.2f} with {above3} counts")

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


class Visualise_Pedestal:

    @staticmethod
    def top_rank_thresholding(indexOI,Nsigma):

        mat_hms, thr1 = mat_minusMean_thr_aboveNsigma(indexOI,Nsigma)

        mat_hms_plus_1, thr2 = mat_minusMean_thr_aboveNsigma(indexOI,Nsigma+1)

        title_1 = f"Thresholded above {Nsigma} sigma"
        title_2 = f"Thresholded below {Nsigma + 1} sigma"
        title_main = f"Top of Image {indexOI} at different thresholding levels: "

        sub_mat1 = mat_hms[0:101,0:101]
        sub_mat2 = mat_hms_plus_1[0:101,0:101]

        plt.figure(figsize=(6, 6))
        plt.suptitle(title_main, fontsize=12, y=0.96 )
        plt.subplot(2, 2, 1), plt.imshow(sub_mat1, cmap="hot"), plt.title(title_1)
        plt.subplot(2, 2, 2), plt.imshow(sub_mat2, cmap="hot"), plt.title(title_2)

        row_sums1 = np.sum(mat_hms, axis=1)
        row_sums2 = np.sum(mat_hms_plus_1, axis=1)
        row_indices = -np.arange(len(row_sums1))

        # Plot for mat_hms
        plt.subplot(2, 2, 3)
        plt.scatter(row_sums1, row_indices, marker='.', color='orange', s=5)
        plt.ylabel("- Row Index")
        plt.xlabel("Total Sum")
        plt.grid(True, linestyle='--', alpha=0.6)

        # Plot for mat_hms_plus_1
        plt.subplot(2, 2, 4)
        plt.scatter(row_sums2, row_indices, marker='.', color='orange',  s=5)
        plt.ylabel("- Row Index")
        plt.xlabel("Total Sum")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def sum_all_images():
        mat_im_sum = np.zeros((2048, 2048))
        folderpath = "Misc_images"
        filename = "summed_all_images_2sigmaTHR.npy"
        filepath = os.path.join(folderpath, filename)

        try:
            mat_im_sum = np.load(filepath)
        except FileNotFoundError:
            for index in range(len(loadData())):
                image_mat, __ = mat_thr_aboveNsigma(index, 2)
                # image_mat = imData[index]
                mat_im_sum += image_mat

            np.save(filepath, mat_im_sum)

        plt.imshow(mat_im_sum, cmap='turbo')
        plt.title("Matrix Sum of All Raw Images")
        plt.ylabel("Image j Index")
        plt.xlabel("Image i Index")
        plt.show()

    @staticmethod
    def plotCompare_Pedestals(list_ordered=reversed([0, 8, 6, 11])):

        bin_edges = np.arange(-40, 225, step=1)

        for index_data in list_ordered:
            print(index_data)
            matIndex_MinusMean = matMinusMean(index_data)
            hist_values, bin_edges = np.histogram(matIndex_MinusMean.flatten(), bins=bin_edges)

            # Plotting the histogram
            plt.hist(
                bin_edges[:-1], bins=bin_edges, weights=hist_values,
                label=f'Image {index_data}',
                alpha=0.8,
                linestyle=['-', '--', '-.', ':'][index_data % 4],  # Vary line style
                linewidth=1.5
            )

        plt.xlabel('ADU Value')
        plt.ylabel('Count')
        plt.title('ADU Histograms with Mean Subtracted')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True)
        plt.yscale('log')
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

def mat_min_mean_thr_above_Nsigma2(matrix,mean,sigma,n_sigma):
    mat_minusMean = matrix.astype(np.int16) - mean
    mat_minusMean[mat_minusMean < 0] = 0

    mat_minusMean_thr = np.where(mat_minusMean > n_sigma*sigma, mat_minusMean, 0)

    return mat_minusMean_thr

def matMinusMean(index_of_interest):
    image_mat = loadData()[index_of_interest]
    # ----------pedestal mean and sigma----------
    ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image_mat, index_of_interest)
    mat_minusMean = image_mat.astype(np.int16) - ped_mean
    # mat_minusMean[mat_minusMean < 0] = 0

    return mat_minusMean


def plot_sigmaThr_mat(indexOI=8,how_many_sigma=2):
    image_mat11, thr2sigma11 = mat_minusMean_thr_aboveNsigma(indexOI,how_many_sigma)
    plt.imshow(image_mat11,cmap='hot')
    plt.show()





if __name__ == "__main__":

    # Visualise_Pedestal().top_rank_thresholding(11,2)

    def plot_reduced_mat(indexOI,NSigma):
        mat_oI, thr = mat_minusMean_thr_aboveNsigma(indexOI,NSigma)

        plt.imshow(mat_oI, cmap='hot')
        plt.show()

    plot_reduced_mat(11,2)


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

    # plotCompare_Pedestals()

    # plot_sigmaThr_mat(11,2)

    def investigate_edge_effects(index_of_interest):

        mat_raw = loadData()[index_of_interest]

        mat_sigma_thr2, thr = mat_minusMean_thr_aboveNsigma(index_of_interest,2)
        mat_sigma_thr3, thr = mat_minusMean_thr_aboveNsigma(index_of_interest, 3)

        # Investigate().printIntenstiy_horizontally(mat_raw, f"{index_of_interest} Raw")
        Investigate().printIntenstiy_horizontally(mat_sigma_thr2, f"{index_of_interest} - 2 sigma thresholded")
        Investigate().printIntenstiy_horizontally(mat_sigma_thr3, f"{index_of_interest} - 3 sigma thresholded")

        # Investigate().printIntenstiy_verticallyWithOutliers(mat_raw, f"{index_of_interest} Raw")
        # Investigate().printIntenstiy_verticallyWithOutliers(mat_sigma_thr3, f"{index_of_interest} - 3 sigma thresholded")


        # Let us now consider the top 500 rows vertically to see if it's about even / if it shows the spectrum properties

        # top_rows = 500
        #
        # mat_raw_top = mat_raw[0:top_rows, :]
        # mat_sigma_thr2_top = mat_sigma_thr2[0:top_rows, :]
        # mat_sigma_thr3_top = mat_sigma_thr3[0:top_rows, :]
        #
        # Investigate.printIntenstiy_vertically(mat_raw_top,f"{index_of_interest} Raw: top {top_rows} rows")
        # Investigate.printIntenstiy_vertically(mat_sigma_thr2_top, f"{index_of_interest} Raw: top {top_rows} rows - 2 sigma thresholded")
        # Investigate.printIntenstiy_vertically(mat_sigma_thr3_top, f"{index_of_interest} Raw: top {top_rows} rows - 3 sigma thresholded")


    # compareCCD_bias(11)

    # create_sum_all_images(False)

    pass

