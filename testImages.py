from imagePreProcessing import *

# TODO: Create unit test of shapes

class TestImages:
    def __init__(self):
        pass

    @staticmethod
    def diagonals():

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

        matCombined[1:4, 1:5] = diag_2pixel
        matCombined[5:9,1:5] = diag_3pixel
        matCombined[10:13,1:4] = diag_1pixel
        matCombined[14:18,1:5] = diag_4pixel
        matCombined[2,5:8] = 70

        return matCombined


    @staticmethod
    def image8_cluster(thr=0):
        topleft = (636,1419)
        bottomright = (650,1427)

        imMat8 = loadData()[8]

        imMat8[imMat8 < thr] = 0

        return imMat8[topleft[0]:bottomright[0],topleft[1]:bottomright[1]]


    @staticmethod
    def image_8_emission_lines(thr=0):
        topleft = (0,1250)
        bottomright = (2047,1650)

        imMat8 = loadData()[8]

        imMat8[imMat8 < thr] = 0

        return imMat8[topleft[0]:bottomright[0],topleft[1]:bottomright[1]]


if __name__ == "__main__":

    # plt.imshow(TestImages().diagonals())
    # plt.show()

    # plt.imshow(TestImages().image8_cluster(80))
    # plt.show()

    pass
