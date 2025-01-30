import numpy as np
import matplotlib.pyplot as plt
import h5py
import itertools
import torch
from torch.nn import AvgPool2d
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

class Investigate():

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

def investigateImageData(index):
    print(type(image_data[index]))
    # np.set_printoptions(threshold=np.inf)
    # print(image_data[index])
    arr = image_data[index]
    print(arr.shape)

def convolvedImage(imageMatrix):
    pool = AvgPool2d(kernel_size=(5,3), padding=(2,1), stride=1)
    return pool(torch.tensor(imageMatrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).reshape(2048, 2048)
# for num in range(0,len(image_data)):
#     print(num)
#     Investigate.showImageAndHist(num)

num = 8
array8Test = image_data[8]
high_intensity_points = np.where(array8Test > 100, array8Test, 0)

imTensor = convolvedImage(high_intensity_points)
# imTensor = convolvedImage(array8Test)
covTest = np.asarray(imTensor)
covTest = np.where(covTest > 0,1,0)
# plt.imshow(covTest,cmap="hot")
# plt.show()
# print(sum(covTest.flatten()))



# Investigate.showImageAndHist(num,200)
# Investigate.showImageIntensityHigherThan(num,100)

