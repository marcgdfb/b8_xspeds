import numpy as np
import matplotlib.pyplot as plt
import h5py
import itertools

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
    def showImageIntensityHigherThan(index,intensity_max=100):
        # Plot a good dataset - here index #8 (but there are others too!)

        arr = image_data[index]
        high_intensity_points = np.where(arr > intensity_max, arr, np.nan)
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


# for num in range(0,len(image_data)):
#     print(num)
#     Investigate.showImageAndHist(num)

# num = 8
# arr = image_data[8]
# arr_large = arr.astype(np.int64)
#
# arr_sum = sum(arr_large.flatten())
#
# print(arr[0,1])
# Investigate.showImageAndHist(num,200)
# Investigate.showImageIntensityHigherThan(num,100)

