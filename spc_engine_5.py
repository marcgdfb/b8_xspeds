from testImages import *
from scipy.optimize import minimize
import math
import os


class SinglePixel:
    def __init__(self, indexOfInterest,how_many_sigma=3):
        self.indexOfInterest = indexOfInterest

        mat_minusMean, thr = mat_minusMean_thr_aboveNsigma(indexOfInterest, how_many_sigma)

        self.imMat = mat_minusMean
        self.rowNum = self.imMat.shape[0]
        self.colNum = self.imMat.shape[1]
        self.image_binary = np.where(self.imMat > 0, 1, 0)


    @staticmethod
    def check_diagonals(convolvedArea, shape_kernel, check_mask):
        pattern_mask = (shape_kernel == 1)
        check_mask = (check_mask == 1)

        # The pattern mask must be perfectly seen
        pattern_match = np.all(convolvedArea[pattern_mask] == 1)
        # The adjacent elements (those defined as 1 in the array) must be 0
        adjacent_match = np.all(convolvedArea[check_mask] == 0)

        if pattern_match and adjacent_match:
            return True
        else:
            return False

    def find_single_pixel_ADU(self):

        list_adu = []

        def single_pixel():
            sp_isolated_kernel = np.array([[0, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 0]])
            sp_check_mask = np.array([[0, 1, 0],
                                      [1, 0, 1],
                                      [0, 1, 0]])

            return {
                "kernels": [sp_isolated_kernel],
                "masks": [sp_check_mask],
            }

        kdict = single_pixel()
        kernels = kdict["kernels"]
        masks = kdict["masks"]

        # Create a matrix that stores checked points
        checked_mat = np.zeros((self.rowNum, self.colNum))

        for kernel, mask in zip(kernels, masks):
            k_rows, k_cols = kernel.shape

            # Convolve the image
            for i in range(self.rowNum - k_rows + 1):
                for j in range(self.colNum - k_cols + 1):

                    if checked_mat[i + 1, j + 1] == 1:
                        continue
                    # Consider areas of the same size as the kernel:
                    convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                    if np.all(convolvedArea == 0):
                        checked_mat[i:i + k_rows, j:j + k_cols] = 1
                        continue

                    if not (np.array_equal(convolvedArea, kernel) or self.check_diagonals(convolvedArea, kernel, mask)):
                        continue

                    singlePixelVal = self.imMat[i + 1, j + 1]

                    checked_mat[i + 1, j + 1] = 1

                    list_adu.append(singlePixelVal)


        return list_adu

    def findSingle_pixel_ADU_no_diagonal(self):

        list_adu = []

        sp_isolated_kernel = np.array([[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]])

        k_rows, k_cols = sp_isolated_kernel.shape
        # Convolve the image
        for i in range(self.rowNum - k_rows + 1):
            for j in range(self.colNum - k_cols + 1):
                # Consider areas of the same size as the kernel:
                convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                if not np.array_equal(convolvedArea, sp_isolated_kernel):
                    continue

                singlePixelVal = self.imMat[i + 1, j + 1]

                list_adu.append(singlePixelVal)

        return list_adu


    def plot_adu_dist(self,bins=200):

        # list_adu = self.find_single_pixel_ADU()
        list_adu = self.findSingle_pixel_ADU_no_diagonal()

        plt.hist(list_adu, bins=bins)
        plt.yscale('log')
        plt.title(f'Histogram of single pixel ADU for image {self.indexOfInterest}')
        plt.xlabel("ADU Value")
        plt.show()


def save_SinglePixel_ADU(list_indices=list_data, folderpath="stored_variables",
                           how_many_sigma=3,plot=False,save=True,ylim=None):

    adu_folderpath = os.path.join(folderpath, "ADU")
    if not os.path.exists(adu_folderpath):
        os.makedirs(adu_folderpath)

    dict_adu_lists = {}

    for idx_oI in list_indices:
        print(idx_oI)
        single_pixelEng = SinglePixel(idx_oI,how_many_sigma)
        adu_list = single_pixelEng.find_single_pixel_ADU()
        dict_adu_lists[idx_oI] = adu_list

        if save:
            np.save(os.path.join(adu_folderpath, f"{idx_oI}.npy"), np.array(adu_list))


    if plot:
        bin_edges = np.arange(0,200,step=1)

        # Plot histograms as lines
        for i, (idx, adu_list) in enumerate(dict_adu_lists.items()):
            counts, _ = np.histogram(adu_list, bins=bin_edges)
            plt.plot(
                bin_edges[:-1], counts,
                label=f'Image {idx}',
                alpha=0.8,
                linestyle=['-', '--', '-.', ':'][i % 4],  # Vary line style
                linewidth=1.5
            )

        plt.xlabel('ADU Value')
        plt.ylabel('Count')
        plt.title(f'ADU Histograms for Single Pixels with {how_many_sigma} sigma thresholding')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        if ylim is not None:
            plt.ylim(top=ylim, bottom=0)
        plt.grid(True)
        plt.show()


def plot_compare_singlePixel_ADU(list_indices=list_data, folderpath="stored_variables",ylim=None):
    dict_adu_lists = {}

    adu_folderpath = os.path.join(folderpath, "ADU")
    for idx_oI in list_indices:
        print(idx_oI)
        adu_array = np.load(os.path.join(adu_folderpath, f"{idx_oI}.npy"))
        dict_adu_lists[idx_oI] = adu_array

    bin_edges = np.arange(0,200,step=1)

    # Plot histograms as lines
    for i, (idx, adu_list) in enumerate(dict_adu_lists.items()):
        counts, _ = np.histogram(adu_list, bins=bin_edges)
        plt.plot(
            bin_edges[:-1], counts,
            label=f'Image {idx}',
            alpha=0.8,
            linestyle=['-', '--', '-.', ':'][i % 4],  # Vary line style
            linewidth=1.5
        )

    plt.xlabel('ADU Value')
    plt.ylabel('Count')
    plt.title('ADU Histograms for Single Pixels')
    plt.legend(loc='upper right', fontsize='small', ncol=2)

    if ylim is not None:
        plt.ylim(top=ylim,bottom=0)
    plt.grid(True)
    plt.show()


class PhotonCounting:
    def __init__(self, indexOfInterest, no_photon_adu_thr=80, sp_adu_thr=150,adu_offset=40, adu_cap=1600,
                 removeRows0To_=0, howManySigma_thr=2, ):
        """
        :param indexOfInterest: Image Matrix index of interest
        :param no_photon_adu_thr: The ADU total lower bound below which we reject the point
        :param sp_adu_thr: The ADU count for a single photon (Inspired by single pixel ADU histogram)
        :param adu_offset: The offset from N * sp_adu_thr which accounts for the deviation from the mean value
        :param adu_cap: A ADU total cap to avoid taking in edge effects or anomalies. The value is chosen, inspired by the anomaly seen in all images
        :param removeRows0To_: Option to remove rows to account for the top edge effects. No significan effect was seen when trialing this
        :param howManySigma_thr: How many sigma away from the mean is thresholded out
        """

        def printVar():
            print("-" * 30)
            print("class PhotonCounting initiated with:")
            print("Index of Interest: ", indexOfInterest)
            print("No Photon ADU Threshold: ", no_photon_adu_thr, "; Single Photon ADU Threshold: ", sp_adu_thr)
            print("ADU offsest: ", adu_offset)
            print("No Photon ADU Upper Threshold: ", adu_cap)
            if removeRows0To_ > 0:
                print("Remove Rows 0 to: ", removeRows0To_)
            print(f"imMat has the mean removed and is then thresholded above {howManySigma_thr} sigma ")

        printVar()

        self.imMatRAW = loadData()[indexOfInterest]
        self.index_of_interest = indexOfInterest

        if removeRows0To_ > 0:
            self.imMatRAW = self.imMatRAW[removeRows0To_:, :]

        self.no_p_adu_thr = no_photon_adu_thr
        self.sp_adu_thr = sp_adu_thr
        self.adu_offset = adu_offset
        self.adu_cap = adu_cap

        def findGaussPedestal_awayFromLines():
            # The i starting index is to avoid the edge effects at the top which is prominent
            iIndexStart, iIndexEnd, jIndexStart, jIndexEnd = 500, 1750, 50, 1150
            matrixOfInterest = self.imMatRAW[iIndexStart:iIndexEnd, jIndexStart:jIndexEnd]
            titleH = f"Image {indexOfInterest} Gaussian Fit for i∊[{iIndexStart},{iIndexEnd}] and j∊[{jIndexStart},{jIndexEnd}] "
            ped8_indexed = Pedestal(matrixOfInterest, titleH, bins=300, pedestalOffset_adu=20, )
            return ped8_indexed.findGaussian(logarithmic=True)

        gaussFitDict = findGaussPedestal_awayFromLines()
        self.meanPedestal = gaussFitDict["mean"][0]  # + gaussFitDict["mean"][1]
        self.sigmaPedestal = gaussFitDict["sigma"][0]  # + gaussFitDict["sigma"][1]

        def removeMeanPedestal():
            mat_minusMean = self.imMatRAW.astype(np.int16) - self.meanPedestal
            mat_minusMean[mat_minusMean < 0] = 0
            return mat_minusMean

        self.imMatMeanRemoved = removeMeanPedestal()
        # Calling this self.imMat as well due to old version
        self.howManySigma = howManySigma_thr
        self.imMat = np.where(self.imMatMeanRemoved > howManySigma_thr * self.sigmaPedestal, self.imMatMeanRemoved, 0)

    def operateOnIslands(self, image_matrix_replace=None, plot_checkedMat=False, diagnosticPrint=False):
        print("-"*30)
        print("OperateOnIslands")
        results_dict = {
            "number_of_islands": 0,
            "number_rejected": 0,
            "number_of_photons": 0,
            "number_higher_than_capture": 0,
            "number_of_points": [],
            "total_ADU": [],
            "list_countij": [],
        }

        if image_matrix_replace is None:
            moI = self.imMat  # Matrix of Interest
        else:
            moI = image_matrix_replace

        nrows, ncols = moI.shape
        # Create a matrix with checked points
        checked_mat = np.zeros((nrows, ncols))

        def scour_for_neighbours(i_idx, j_idx, island_list, islandValList):
            if i_idx < 0 or i_idx >= nrows or j_idx < 0 or j_idx >= ncols:
                return
            if checked_mat[i_idx, j_idx] == 1 or moI[i_idx, j_idx] == 0:
                return
            if len(island_list) > 100:
                return

            checked_mat[i_idx, j_idx] = 1
            island_list.append((i_idx, j_idx))
            islandValList.append(moI[i_idx, j_idx])

            # Now scour for neighbours

            for di, dj in [(-1, 0),
                           (0, -1), (0, 1),
                           (1, 0), ]:
                scour_for_neighbours(i_idx + di, j_idx + dj, island_list, islandValList)

        for i in range(nrows):
            for j in range(ncols):
                if moI[i, j] != 0 and checked_mat[i, j] == 0:
                    island = []
                    islandVals = []
                    scour_for_neighbours(i, j, island, islandVals)  # island has been appended and totVal acquired

                    totVal = 0
                    for val in islandVals:
                        totVal += val

                    numPoints = len(island)

                    results_dict["number_of_islands"] += 1
                    results_dict["number_of_points"].append(numPoints)
                    results_dict["total_ADU"].append(totVal)

                    if totVal < self.no_p_adu_thr:
                        results_dict["number_rejected"] += 1
                        continue
                    elif totVal > self.adu_cap:
                        print(f"There was more than the cap i~{i}, j~{j}, totVal = {totVal}")
                        results_dict["number_higher_than_capture"] += 1
                        continue
                    else:
                        # TotVal < N_photons * single  + offset
                        lessThanN = (totVal - self.adu_offset) / self.sp_adu_thr
                        numPhotons = math.ceil(lessThanN)

                    results_dict["number_of_photons"] += numPhotons

                    # print(totVal)

                    idx_ordered_list = sorted(range(len(islandVals)), key=lambda i: islandVals[i], reverse=True)

                    if numPhotons > numPoints:
                        idx_max = idx_ordered_list[0]
                        ij_tuple = island[idx_max]

                        results_dict["list_countij"].append([numPhotons, ij_tuple[0], ij_tuple[1]])
                    else:
                        for number in range(numPhotons):
                            idx_photon = idx_ordered_list[number]  # ie find the idx of this photon
                            ij_tuple = island[idx_photon]

                            if diagnosticPrint:
                                print(f"Photon at (i,j) = {ij_tuple}, ADU = {moI[ij_tuple[0], ij_tuple[1]]}")


                            results_dict["list_countij"].append([1, ij_tuple[0], ij_tuple[1]])

        print("-" * 30)
        print("Number of islands: ", results_dict["number_of_islands"])
        print("Number rejected: ", results_dict["number_rejected"])
        print("Number of counts: ", results_dict["number_of_photons"])
        print("Number of ADU counts higher than capture: ", results_dict["number_higher_than_capture"])

        if plot_checkedMat:
            plt.imshow(checked_mat)
            plt.show()

        return results_dict


class Unit_testing:

    @staticmethod
    def unitTest1(fillFrac=0.1, matrix_size=(2048, 2048), mean_adu=150, std_adu=10,
                  returnJustImage=False, seed=125,plotMat=False):
        spcTrain = SPC_Train_images(2)

        unit_test_mat = spcTrain.createTestData(fillfraction=fillFrac, matrix_size=matrix_size, mean_adu=mean_adu,
                                                std_adu=std_adu,
                                                returnJustImage=returnJustImage, seed=seed)

        num_photons = fillFrac * matrix_size[0] * matrix_size[1]

        if plotMat:
            plt.imshow(unit_test_mat)
            plt.title(f"Unit test with fill fraction {fillFrac} and seed {seed}")
            plt.show()

        adu_thr_guess = np.array([
            80,  # no_photon_adu_thr
            180,  # sp_adu_thr
            30,  # ADU_offset
        ])

        def loss_function(params):
            pc_eng = PhotonCounting(8, no_photon_adu_thr=params[0], sp_adu_thr=params[1],adu_offset=params[2], adu_cap=1650,
                                    removeRows0To_=0, howManySigma_thr=2, )

            results_dict = pc_eng.operateOnIslands(image_matrix_replace=unit_test_mat, diagnosticPrint=False)

            numCaptured = results_dict["number_of_photons"]
            num_above_capture = results_dict["number_higher_than_capture"]


            loss = (num_photons - numCaptured)**2

            return loss


        result = minimize(loss_function, adu_thr_guess, method='Nelder-Mead', options={'maxiter': 30})

        print(result.x)

    @staticmethod
    def test_im8clusters():
        mat_cluster_1 = TestImages().image8_cluster_reducedScheme1()
        mat_cluster_2 = TestImages().image8_cluster_reducedScheme2()
        mat_cluster_3 = TestImages().image8_cluster_reducedScheme3()

        pc_eng = PhotonCounting(8,
                                removeRows0To_=0, howManySigma_thr=2, )

        print("Cluster1")
        results_dict = pc_eng.operateOnIslands(image_matrix_replace=mat_cluster_1, diagnosticPrint=True)
        print("Cluster2")
        results_dict = pc_eng.operateOnIslands(image_matrix_replace=mat_cluster_2, diagnosticPrint=True)
        print("Cluster3")
        results_dict = pc_eng.operateOnIslands(image_matrix_replace=mat_cluster_3, diagnosticPrint=True)

        image8ClusterShow()


if __name__ == "__main__":

    # plot_compare_singlePixel_ADU(ylim=1000)


    # save_SinglePixel_ADU(plot=True,how_many_sigma=1,ylim=1000,save=False)

    # Unit_testing().unitTest1(fillFrac=0.1,matrix_size=(100,100),plotMat=True)

    # Unit_testing().test_im8clusters()

    def CheckAllSinglePixelHist(indices_of_interest=list_data):
        print("Checking all single-pixel histograms")
        for index_ in indices_of_interest:
            SinglePixel(index_,2).plot_adu_dist(bins=200)


    # CheckAllSinglePixelHist([16])


    def compareFaveLittleSpot():

        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=2)
        sigma = pc.sigmaPedestal
        mean = pc.meanPedestal

        thr_sigma_1 = 2
        thr_sigma_2 = 3

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(
            TestImages().image8_cluster(thr_after_mean_removed=thr_sigma_1 * sigma, mean=mean), cmap='hot'), plt.title(
            f"Image Thresholded above {thr_sigma_1} sigma")
        plt.subplot(1, 2, 2), plt.imshow(
            TestImages().image8_cluster(thr_after_mean_removed=thr_sigma_2 * sigma, mean=mean), cmap='hot'), plt.title(
            f"Image 8 Thresholded above {thr_sigma_2} sigma")
        plt.show()


    # compareFaveLittleSpot()





    pass
