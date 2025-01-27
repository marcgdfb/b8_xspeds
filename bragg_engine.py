import pandas as pd

from tools import *

class Bragg:
    def __init__(self,imageMatrix):
        """

        :param imageMatrix: Numpy array containing image
        """
        self.imageMatrix = imageMatrix
        self.x_pixels = self.imageMatrix.shape[1]
        self.y_pixels = self.imageMatrix.shape[0]



    def simpleWithDarkImage(self):

        # Without using single photon counting and retaining dark images the most logical way
        # to try and get a spectrum for Bragg's is to divide the intensity by the expected E value

        # TODO: Add Energy max,min count max min i.e uncertainty.
        df_spectrum = pd.DataFrame({'Energy': [],
                                    'Count': []
                                    })

        for i in range(self.imageMatrix.shape[0]):
            for j in range(self.imageMatrix.shape[1]):
                E,Emax,Emin = xypixel_observationPlane_to_energy(j,i)

                normalised_count = self.imageMatrix[i,j] / E


                df_new = pd.DataFrame({'Energy': [E],
                                       'Count': [normalised_count]})

                df_spectrum = pd.concat([df_spectrum, df_new], ignore_index=True)



    # TODO: remember to include fano factor for intensity of image element