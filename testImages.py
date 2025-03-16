import numpy as np
from constants import *
from imagePreProcessing import *

# TODO: Create more unit tests

def generate_noisy_ellipse_Matrix(C, A, y0, B, noise_level=2.0):
    y_values = np.arange(start=0, stop=length_detector_pixels, step=1)
    x_values = np.array([
        C + A - A * np.sqrt(1 - (y - y0) ** 2 / B ** 2) if abs(y - y0) <= B else np.nan
        for y in y_values
    ])
    matrix = np.zeros((2048, 2048))
    noise = np.random.normal(0, noise_level, size=(2048, 2048))
    x_indices = np.clip(np.round(x_values).astype(int), 0, 2047)
    y_indices = np.clip(np.round(y_values).astype(int), 0, 2047)
    matrix[x_indices, y_indices] = 1
    matrix += noise
    return matrix

class TestImages:
    def __init__(self):
        pass

    def diagonals(self):

        matCombined = np.zeros((20,20))

        diag_1pixel = np.array([[0, 0, 50],
                                [0, 150, 0],
                                [0, 0, 0]])

        diag_2pixel = np.array([[0, 0, 0, 50],
                                [0, 90, 80, 0],
                                [0, 0, 0, 0]])

        diag_3pixel = np.array([[0, 0, 0, 0],
                                [0, 70, 80, 0],
                                [50, 0, 70, 0],
                                [0, 0, 0, 0]])

        diag_4pixel = np.array([[0, 0, 0, 50],
                                [0, 60, 60, 0],
                                [0, 70, 60, 0],
                                [0, 0, 0, 0]])

        diag_5pixel = np.array([[0, 0, 0, 0, 150],
                                [0, 50, 50, 0, 0],
                                [0, 60, 50, 40, 0],
                                [0, 0, 0, 0, 0]])

        long_l = np.array([[0, 0, 0, 0, 0],
                            [0, 50, 0, 0, 0],
                            [0, 60, 50, 40, 0],
                            [0, 0, 0, 0, 0]])

        t_junc = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 50, 0, 0],
                            [0, 60, 50, 40, 0],
                            [0, 0, 0, 0, 0]])

        zigzag = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 50, 40, 0],
                            [0, 60, 50, 0, 0],
                            [0, 0, 0, 0, 0]])

        matCombined[1:4, 1:5] = diag_2pixel
        matCombined[5:9,1:5] = diag_3pixel
        matCombined[10:13,1:4] = diag_1pixel
        matCombined[14:18,1:5] = diag_4pixel
        matCombined[2,5:8] = 70   # Create the line

        matCombined[5:9,6:11] = diag_5pixel
        matCombined[10:14,6:11] = long_l
        matCombined[15:19,6:11] = t_junc

        matCombined[1:15,11:19] = self.image8_cluster(thr_after_mean_removed=20, mean=60)
        matCombined[16:20,12:17] = zigzag


        return matCombined

    def islands(self):
        matCombined = np.zeros((20,9))
        matCombined[0:14,0:8] = self.image8_cluster(thr_after_mean_removed=2*9.5, mean=58.598)
        # matCombined[15:20,0:5] = np.array([[0,0,100,0,0],
        #                                    [0,0,100,50,30],
        #                                    [0,50,100,60,0],
        #                                    [0,100,100,80,70],
        #                                    [0,0,0,0,0],]
        #                                   )


        return matCombined

    @staticmethod
    def image8_cluster(thr_after_mean_removed=0.0, mean=0.0):
        topleft = (636,1419)
        bottomright = (650,1427)

        imMat8 = loadData()[8]

        mat_minusMean = imMat8.astype(np.int16) - mean
        mat_minusMean[mat_minusMean < thr_after_mean_removed] = 0

        return mat_minusMean[topleft[0]:bottomright[0],topleft[1]:bottomright[1]]

    @staticmethod
    def image_8_emission_lines(thr=0.0):
        topleft = (0,1250)
        bottomright = (2047,1650)

        imMat8 = loadData()[8]

        imMat8[imMat8 < thr] = 0

        return imMat8[topleft[0]:bottomright[0],topleft[1]:bottomright[1]]

    @staticmethod
    def image_8_post_kernels():
        im8_postk = np.load(r"old_logs_and_stored_variables/v1/data_logs\image_matrices\image_8\test3_final.npy")
        return im8_postk


if __name__ == "__main__":

    # plt.imshow(TestImages().islands())
    # plt.show()

    # plt.imshow(TestImages.image_8_post_kernels())
    # plt.show()

    # plt.imshow(np.load(r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\image_matrices\image_8\test2_3_3.npy"))
    # plt.show()

    # plt.imshow(TestImages().image8_cluster(80))
    # plt.show()
    #
    # plt.imshow(TestImages().image_8_emission_lines(80))
    # plt.show()


    pass
