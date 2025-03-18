import numpy as np
from constants import *
from pedestal_engine_v2 import *
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
    def image8_cluster_reducedScheme1():

        mat8, thr = mat_minusMean_thr_aboveNsigma(8,how_many_sigma=2)

        topleft = (636, 1419)
        bottomright = (650, 1427)

        return mat8[topleft[0]:bottomright[0], topleft[1]:bottomright[1]]

    @staticmethod
    def image8_cluster_reducedScheme2():

        mat8, thr = mat_minusMean_thr_aboveNsigma(8,how_many_sigma=2)

        topleft = (660, 1420)
        bottomright = (676, 1427)

        return mat8[topleft[0]:bottomright[0], topleft[1]:bottomright[1]]

    @staticmethod
    def image8_cluster_reducedScheme3():
        mat8, thr = mat_minusMean_thr_aboveNsigma(8, how_many_sigma=2)

        topleft = (1760, 1472)
        bottomright = (1781, 1481)

        return mat8[topleft[0]:bottomright[0], topleft[1]:bottomright[1]]

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


class SPC_Train_images:
    def __init__(self,how_many_sigma, indexOfInterest=8,):
        # using image 8
        self.indexOfInterest = indexOfInterest
        image8_mat = loadData()[indexOfInterest]

        ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image8_mat, indexOfInterest)

        image8_mat_minusMean, thr = mat_minusMean_thr_aboveNsigma(indexOfInterest,how_many_sigma)

        self.ped_sigma = ped_sigma
        self.imMat = image8_mat_minusMean

    def get_cluster(self,topLeftCorner_ij, bottomRightCorner_ij):
        return self.imMat[topLeftCorner_ij[0]:bottomRightCorner_ij[0]+1, topLeftCorner_ij[1]:bottomRightCorner_ij[1]+1]

    def testData1(self):
        return self.get_cluster(topLeftCorner_ij=(1419,636), bottomRightCorner_ij=(0,0))

    def testData2(self):
        return self.get_cluster(topLeftCorner_ij=(1457,1629), bottomRightCorner_ij=(1460,1632))

    def anomalyData(self):
        clusterMat = self.get_cluster(topLeftCorner_ij=(1391,1753), bottomRightCorner_ij=(1400,1761))
        total = clusterMat.sum()
        print(f"Anomalous Cluster total for image {self.indexOfInterest} is {total}")

        return clusterMat

    def createTestData(self,fillfraction=0.1, matrix_size=(100, 100), mean_adu=150, std_adu=30,
                       returnJustImage=False,seed=None,addAnomaly=True ):

        num_photons = int(matrix_size[0] * matrix_size[1] * fillfraction)
        if seed is not None:
            np.random.seed(seed)

        image = np.zeros(matrix_size, dtype=float)
        noise_mat = generate_pedestal_mat(self.ped_sigma,x0=0,nrows=matrix_size[0],ncols=matrix_size[1])

        # For reproducibility in tests, you might set a random seed outside this function if desired.
        spread_patterns = [
            [(0, 0)],  # Single pixel
            [(0, 0)],  # Single pixel (again so we don't over favour the others)
            [(0, 0), (0, 1)],  # Adjacent horizontally
            [(0, 0), (1, 0)],  # Adjacent vertically
            [(0, 0), (0, 1), (1, 0)],  # L-shape 1
            [(0, 0), (0, 1), (1, 1)],  # L-shape 2
            [(0, 0), (1, 1), (1, 0)],  # L-shape 3
            [(0, 1), (1, 1), (1, 0)],  # L-shape 4
            [(0, 0), (0, 1), (1, 0), (1, 1)],  # Square spread
            [(0, 0), (0, 1), (1, 0), (1, 1)]  # Square spread (again)
        ]
        weights = [
            [1.0],  # Single pixel
            [1.0],  # Single pixel
            [0.7, 0.3],  # Adjacent horizontally
            [0.7, 0.3],  # Adjacent vertically
            [0.6, 0.2, 0.2],  # L-shape 1, brightest at top corner
            [0.2, 0.6, 0.2],  # L-shape 2,
            [0.2, 0.2, 0.6],  # L-shape 3,
            [0.2, 0.6, 0.2],  # L-shape 4,
            [0.4, 0.3, 0.2, 0.1],  # Square spread, brightest at top-left
            [0.1, 0.3, 0.2, 0.4]  # Square spread, brightest at bottom right
        ]
        for _ in range(num_photons):
            # Choose random base pixel
            base_x = np.random.randint(0, matrix_size[0])
            base_y = np.random.randint(0, matrix_size[1])

            # Choose a spread pattern
            pattern_index = np.random.randint(0, len(spread_patterns))
            pattern = spread_patterns[pattern_index]
            pattern_weights = weights[pattern_index]
            adu_signal = np.random.normal(loc=mean_adu, scale=std_adu)

            # Normalize weights to ensure total ADU is preserved
            normalized_weights = np.array(pattern_weights) / sum(pattern_weights)

            for (dx, dy), weight in zip(pattern, normalized_weights):
                x = np.clip(base_x + dx, 0, matrix_size[0] - 1)
                y = np.clip(base_y + dy, 0, matrix_size[1] - 1)
                image[x, y] += adu_signal * weight

        imageWithNoise = image + noise_mat
        imageWithNoise = np.where(imageWithNoise > 2*self.ped_sigma,imageWithNoise,0)

        if returnJustImage:
            return image
        elif addAnomaly:
            anomalyMat = self.anomalyData()

            if matrix_size == (2048,2048):
                imageWithNoise[1392:1401, 1754:1762] = anomalyMat
            else:
                imageWithNoise[72:81,54:62] = anomalyMat

            return imageWithNoise

        else:
            return imageWithNoise


def test_createTestData(fillFraction=0.1, matrix_size=(100, 100), mean_adu=150, std_adu=10,
                   returnJustImage=False,seed=125,addAnomaly=True):
    spcTrain = SPC_Train_images(2)

    mat = spcTrain.createTestData(fillfraction=fillFraction, matrix_size=matrix_size, mean_adu=mean_adu, std_adu=std_adu,
                   returnJustImage=returnJustImage,seed=seed,addAnomaly=addAnomaly)

    plt.imshow(mat)
    plt.title(f"Unit Test Image with noise and 2 sigma thresholding and Anomaly. Seed = {seed}")
    plt.show()

    return mat


def image8ClusterShow():
    mat_cluster_1 = TestImages().image8_cluster_reducedScheme1()
    mat_cluster_2 = TestImages().image8_cluster_reducedScheme2()
    mat_cluster_3 = TestImages().image8_cluster_reducedScheme3()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # Colorbar position

    # Get global min and max for consistent color scaling
    vmin = min(mat_cluster_1.min(), mat_cluster_2.min(), mat_cluster_3.min())
    vmax = max(mat_cluster_1.max(), mat_cluster_2.max(), mat_cluster_3.max())

    # Plot each matrix
    im1 = axs[0].imshow(mat_cluster_1, vmin=vmin, vmax=vmax, cmap='viridis')
    # axs[0].set_title("Scheme 1")
    im2 = axs[1].imshow(mat_cluster_2, vmin=vmin, vmax=vmax, cmap='viridis')
    # axs[1].set_title("Scheme 2")
    im3 = axs[2].imshow(mat_cluster_3, vmin=vmin, vmax=vmax, cmap='viridis')
    # axs[2].set_title("Scheme 3")

    # Shared colorbar
    fig.colorbar(im1, cax=cbar_ax)

    # Global title
    fig.suptitle("Unit Tests: Image 8", fontsize=16)

    plt.show()



if __name__ == "__main__":

    # test_createTestData(0.1,matrix_size=(2048,2048), )


    def findAnomalyTotal():
        for index_ in [19]:
            spcTest = SPC_Train_images(2,index_)
            mat_anomaly = spcTest.anomalyData()

            plt.imshow(mat_anomaly)
            plt.show()


    findAnomalyTotal()

    # spc8 = SPC_Train_images(2,8)
    # plt.imshow(spc8.get_cluster(topLeftCorner_ij=(500,500), bottomRightCorner_ij=(520,520)))
    # plt.show()

    # plt.imshow(TestImages().islands())
    # plt.show()

    # plt.imshow(TestImages.image_8_post_kernels())
    # plt.show()

    # plt.imshow(np.load(r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\image_matrices\image_8\test2_3_3.npy"))
    # plt.show()




    image8ClusterShow()


    pass
