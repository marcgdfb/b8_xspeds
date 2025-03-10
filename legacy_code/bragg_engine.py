import pandas as pd

from tools import *

class Visualise:

    @staticmethod
    def spectrum(df,energyCol="Energy",countIntensityCol="Count"):

        sns.lineplot(data=df,x=energyCol, y=countIntensityCol)
        plt.show()


class Bragg:
    def __init__(self,imageMatrix,pixelWidth=pixel_width):
        """

        :param imageMatrix: Numpy array containing image
        """
        self.imageMatrix = imageMatrix
        self.pixelWidth = pixelWidth
        self.x_pixels = self.imageMatrix.shape[1]
        self.y_pixels = self.imageMatrix.shape[0]



    def simpleWithDarkImage(self):

        # Without using single photon counting and retaining dark images the most logical way
        # to try and get a spectrum for Bragg's is to divide the intensity by the expected E value

        df_spectrum = pd.DataFrame({'Energy': [],
                                    'Count': []
                                    })

        for i in range(self.imageMatrix.shape[0]):
            for j in range(self.imageMatrix.shape[1]):
                E,Emax,Emin = xypixel_observationPlane_to_energy(j,i,num_pixels_x=self.x_pixels,num_pixels_y=self.y_pixels)

                normalisation_solid = solidAngleNormalisation(E,widthDetector=(self.y_pixels * pixel_width))
                print(i,j,normalisation_solid)
                normalised_count = self.imageMatrix[i,j] / (E*normalisation_solid)



                df_new = pd.DataFrame({'Energy': [E],
                                       'Count': [normalised_count]})

                df_spectrum = pd.concat([df_spectrum, df_new], ignore_index=True)


        # Grouping values of same Energy

        df_spectrum_grouped = pd.DataFrame(df_spectrum.groupby("Energy")["Count"].sum())

        Visualise.spectrum(df_spectrum_grouped)

