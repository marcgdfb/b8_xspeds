import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Some of the code here is from ideas of finding the probability that a shape arises from the noise.
# It was decided to prioritise accuracy and hence threshold out most of this instead, making some of the starting
# ideas here redundant. Though findExpectedCountsAbove123sigma is still used

def gaussian(x,A,x0,sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def integrate_gaussian(lowerBound,upperBound,A,x0,sigma):
    return quad(gaussian,lowerBound,upperBound,args=(A,x0,sigma))[0]

def prob_gaussian_above_x(x,x0,sigma):
    """
    :param x:
    :param x0:
    :param sigma:
    :return:
    """

    return 0.5 * (1 - math.erf((x - x0) / (math.sqrt(2) * sigma)))

def findExpectedCountsAbove123sigma(gaussFitDict,npixels):
    """

    :param gaussFitDict: dictionary with keys "mean", "amplitude","sigma","x_1sigma","x_2sigma","x_3sigma"
    Each value of this dictionary is a list with the 0th value being the expected value and the 1st its uncertainty (again a sigma like value)
    :param npixels: number of pixels in this image
    :return:
    """

    # Could add here something to find the extreme values of these counts

    x0 = gaussFitDict["mean"][0]
    d_x0 = gaussFitDict["mean"][1]
    amp = gaussFitDict["amplitude"][0]
    d_amp = gaussFitDict["amplitude"][1]
    sigma = gaussFitDict["sigma"][0]
    d_sigma = gaussFitDict["sigma"][1]

    count_above_list = []

    for N in [1,2,3]:
        probabilityAboveX = prob_gaussian_above_x(x=x0+N*sigma,x0=x0,sigma=sigma)
        expectedCount = npixels * probabilityAboveX
        count_above_list.append(expectedCount)

    return count_above_list

def generateImageGauss(x0,sigma,nrows=2048,ncols=2048,printMatrix=False,threshold_num_sigma=0):
    matGauss = np.zeros((nrows,ncols))

    for i in range(nrows):
        for j in range(ncols):
            matGauss[i,j] = np.random.normal(loc=x0,scale=sigma)

    if threshold_num_sigma > 0:
        matGauss[matGauss < x0 + threshold_num_sigma * sigma] = 0

    if printMatrix:
        title = f"Generated Image using random gaussian filtered above {threshold_num_sigma} sigma"
        plt.imshow(matGauss)
        plt.title(title)
        plt.show()

    return matGauss


def generate_pedestal_mat(sigma, x0=0, nrows=2048, ncols=2048):
    matGauss = np.zeros((nrows,ncols))

    for i in range(nrows):
        for j in range(ncols):
            matGauss[i,j] = np.random.normal(loc=x0,scale=sigma)

    return matGauss





if __name__ == "__main__":

    def test_probabilityOfStructures():

        amp = 81600
        mean = 0  # Assuming that we have removed the mean from it
        sigma = 8.9

        countTotal = integrate_gaussian(lowerBound=-np.inf,upperBound=np.inf,A=amp,x0=mean,sigma=sigma)

        # The number of pixels used for this was between i∊[500,1750] and j∊[50,1150]
        print("Number of pixels", (1750-500)*(1150-50))
        print("total Count from Gauss",countTotal)

        prob_above30 = prob_gaussian_above_x(x=30,x0=mean,sigma=sigma)

    # test_probabilityOfStructures()

    # generateImageGauss(x0=0,sigma=8.9,nrows=2048,ncols=2048,printMatrix=True,threshold_num_sigma=3)



    pass

