from legacy_code.getImageData import *
from scipy.optimize import curve_fit

#region growth

class Pedastal:
    """
    Look at dark images,
    See how the edge effects are important
    gaussian fit to pedastal --> cutoff = where it hits x axis
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
            pedastal_values = hist_values[:maxAmpIndex + self.pedstalCutoff]
            pedastal_centers = bin_centers[:maxAmpIndex + self.pedstalCutoff]
            bar_widths = np.diff(bin_edges[:maxAmpIndex + self.pedstalCutoff + 1])

            plt.figure(figsize=(10,5))

            # Raw Hist
            plt.subplot(1,2,1)
            plt.hist(self.imageMatrix.flatten(), self.bins)
            plt.yscale('log')
            plt.title(f"{self.title_matrix}")

            # Hist with cutoff
            plt.subplot(1, 2, 2)
            plt.bar(x=pedastal_centers,height=pedastal_values,width=bar_widths,align='center')
            plt.show()
        else:
            plt.hist(self.imageMatrix.flatten(), self.bins)
            plt.yscale('log')
            plt.title(f"{self.title_matrix}")
            plt.show()

    def findGaussian(self):

        # Obtain Histogram Data, the function gives the edges of the bin
        hist_values, bin_edges = np.histogram(self.imageMatrix.flatten(), bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # print(hist_values)

        # Finding the peak x,y values
        # This assumes the bins are not too thin such that we get distorted results
        maxAmp = max(hist_values)
        maxAmpIndex = np.argmax(hist_values)
        maxAmpBin = bin_centers[maxAmpIndex]

        binRight_edge = bin_centers[3*maxAmpIndex]

        print("maxAmp",maxAmp)
        print("maxAmpIndex",maxAmpIndex)

        # Limiting the data to the peak

        if self.pedstalCutoff is not None:

            pedastal_values = hist_values[:maxAmpIndex + self.pedstalCutoff]
            pedastal_centers = bin_centers[:maxAmpIndex + self.pedstalCutoff]

        def gaussianFunction(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        params_initial = [
            maxAmp,     # a
            maxAmpBin,     # x0
            5     # sigma
        ]

        params, covariance = curve_fit(gaussianFunction, pedastal_centers, pedastal_values, p0=params_initial)

        xvals_fitted = np.linspace(min(pedastal_centers), binRight_edge, 100)
        yvals_fitted = gaussianFunction(xvals_fitted, *params)

        plt.bar(bin_centers, hist_values, width=np.diff(bin_edges), alpha=0.6, label="Histogram")
        plt.plot(xvals_fitted, yvals_fitted, 'r-', label="Gaussian Fit")
        plt.yscale("log")  # Match the original scale if needed
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.ylim(1)
        plt.legend()
        plt.title(self.title_matrix + f" bins={self.bins}, fitted until peak + {self.pedstalCutoff} indices")
        plt.show()


ped8 = Pedastal(array8Test,"Image 8",bins=150,pedstalCutoffOffset=15)

ped8.findGaussian()
# ped8.printHistogram()
