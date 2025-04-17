from testImages import *
from scipy.optimize import minimize
import os
import time
from tools import *


def spc_folder(folderpath_, index_of_interest_):
    index_folder = os.path.join(folderpath_, str(index_of_interest_))
    spc_folderpath = os.path.join(index_folder, "spc")
    if not os.path.exists(spc_folderpath):
        os.makedirs(spc_folderpath)
    return spc_folderpath


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


class SinglePixel:
    def __init__(self, indexOfInterest, how_many_sigma=3):
        self.indexOfInterest = indexOfInterest

        mat_minusMean, thr = mat_minusMean_thr_aboveNsigma(indexOfInterest, how_many_sigma)

        self.imMat = mat_minusMean
        self.rowNum = self.imMat.shape[0]
        self.colNum = self.imMat.shape[1]
        self.image_binary = np.where(self.imMat > 0, 1, 0)

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

                    if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
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

    def photon_count(self, no_photon_adu_thr=80, sp_adu_thr=150, adu_offset=40, adu_cap=1600):

        list_count_ij = []
        num_rejected = 0
        count_photons = 0

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

                    if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                        continue

                    singlePixelVal = self.imMat[i + 1, j + 1]

                    checked_mat[i + 1, j + 1] = 1

                    if singlePixelVal < no_photon_adu_thr:
                        num_rejected += 1
                        continue
                    elif singlePixelVal > adu_cap:
                        print(f"There was more than the cap i~{i}, j~{j}, totVal = {singlePixelVal}")
                        continue
                    else:
                        # TotVal < N_photons * single  + offset
                        lessThanN = (singlePixelVal - adu_offset) / sp_adu_thr
                        # numPhotons = math.ceil(lessThanN)
                        numPhotons = 1
                        count_photons += numPhotons

                    list_count_ij.append([numPhotons, i + 1, j + 1])

        print("Number of Single Pixel Hits Rejected: ", num_rejected)
        print("Number of Single Pixel Photons found: ", count_photons)

        return list_count_ij

    def plot_adu_dist(self, bins=200):

        # list_adu = self.find_single_pixel_ADU()
        list_adu = self.findSingle_pixel_ADU_no_diagonal()

        plt.hist(list_adu, bins=bins)
        plt.yscale('log')
        plt.title(f'Histogram of single pixel ADU for image {self.indexOfInterest}')
        plt.xlabel("ADU Value")
        plt.show()


def save_SinglePixel_ADU(list_indices=list_good_data, folderpath="stored_variables",
                         how_many_sigma=3, plot=False, save=True, ylim=None):
    adu_folderpath = os.path.join(folderpath, "ADU")
    if not os.path.exists(adu_folderpath):
        os.makedirs(adu_folderpath)

    dict_adu_lists = {}

    for idx_oI in list_indices:
        print(idx_oI)
        single_pixelEng = SinglePixel(idx_oI, how_many_sigma)
        adu_list = single_pixelEng.find_single_pixel_ADU()
        dict_adu_lists[idx_oI] = adu_list

        if save:
            np.save(os.path.join(adu_folderpath, f"{idx_oI}.npy"), np.array(adu_list))

    if plot:
        bin_edges = np.arange(0, 200, step=1)

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


def plot_compare_singlePixel_ADU(list_indices=list_good_data, folderpath="stored_variables", ylim=None):
    dict_adu_lists = {}

    adu_folderpath = os.path.join(folderpath, "ADU")
    for idx_oI in list_indices:
        print(idx_oI)
        adu_array = np.load(os.path.join(adu_folderpath, f"{idx_oI}.npy"))
        dict_adu_lists[idx_oI] = adu_array

    bin_edges = np.arange(0, 200, step=1)

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
        plt.ylim(top=ylim, bottom=0)
    plt.grid(True)
    plt.show()


class Island_PhotonCounting:
    def __init__(self, indexOfInterest, no_photon_adu_thr=100, sp_adu_thr=150, adu_offset=30, adu_cap=5000,
                 removeRows0To_=0, howManySigma_thr=2, how_many_more_sigma=2,
                 diagnostic_print=False,declareVars=True):
        """
        :param indexOfInterest: Image Matrix index of interest
        :param no_photon_adu_thr: The ADU total lower bound below which we reject the point
        :param sp_adu_thr: The ADU count for a single photon (Inspired by single pixel ADU histogram)
        :param adu_offset: The offset from N * sp_adu_thr which accounts for the deviation from the mean value
        :param adu_cap: A ADU total cap to avoid taking in edge effects or anomalies. The value is chosen, inspired by the anomaly seen in all images
        :param removeRows0To_: Option to remove rows to account for the top edge effects. No significan effect was seen when trialing this
        :param howManySigma_thr: How many sigma away from the mean is thresholded out
        :param how_many_more_sigma: How many more sigma we threshold upon having found an island
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

        if declareVars:
            printVar()

        imMatRAW = loadData()[indexOfInterest]

        # Removing anomalies from image
        imMat_processed = removeAnomaly(imMatRAW)
        # Removing Lines at left most edge
        imMat_processed[:, 0:3] = 0

        self.index_of_interest = indexOfInterest

        if removeRows0To_ > 0:
            imMat_processed = imMatRAW[removeRows0To_:, :]

        self.no_p_adu_thr = no_photon_adu_thr
        self.sp_adu_thr = sp_adu_thr
        self.adu_offset = adu_offset
        self.adu_cap = adu_cap

        meanPedestal, sigmaPedestal = pedestal_mean_sigma_awayFromLines(imMat_processed,
                                                                        indexOfInterest=indexOfInterest)
        self.meanPedestal = meanPedestal
        self.sigmaPedestal = sigmaPedestal

        def removeMeanPedestal():
            mat_minusMean = imMat_processed.astype(np.int16) - self.meanPedestal
            mat_minusMean[mat_minusMean < 0] = 0
            return mat_minusMean

        self.imMatMeanRemoved = removeMeanPedestal()
        # Calling this self.imMat as well due to old version
        self.howManySigma = howManySigma_thr

        self.imMat = np.where(self.imMatMeanRemoved > howManySigma_thr * self.sigmaPedestal, self.imMatMeanRemoved, 0)
        # Apply extra sigma threshold to the top 500 rows
        self.imMat[:500] = np.where(
            self.imMatMeanRemoved[:500] > (1 + self.howManySigma) * self.sigmaPedestal,
            self.imMatMeanRemoved[:500], 0
        )

        self.how_many_more_sigma = how_many_more_sigma
        self.diagnostic_print = diagnostic_print

        # plt.imshow(self.imMat)
        # plt.show()

    def operateOnIslands(self, image_matrix_replace=None, plot_checkedMat=False,
                         diagnostic_of_large_islands=False, unit_test_correct_mat=None,
                         Report=True):

        print("-" * 30)
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

        noise_dominated_islands = 0

        dict_multiphoton_counts = {}

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

                    min_i = min(i for i, j in island)
                    max_i = max(i for i, j in island)
                    min_j = min(j for i, j in island)
                    max_j = max(j for i, j in island)

                    # Creating mini matrices with the island in it:
                    matrix_island = np.zeros((max_i - min_i + 1, max_j - min_j + 1))

                    # Fill the matrix with the corresponding values
                    for (i_, j_), val in zip(island, islandVals):
                        matrix_island[i_ - min_i, j_ - min_j] = val

                    totVal = 0
                    for val in islandVals:
                        totVal += val

                    numPoints = len(island)

                    # use the island vals more inteligently

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

                        # Want to penalise very large islands:
                        pixel_size_island_mat = matrix_island.shape[0] * matrix_island.shape[1]
                        penalty = 0.0533 * self.sigmaPedestal * pixel_size_island_mat

                        totVal -= penalty

                        lessThanN = totVal / self.sp_adu_thr
                        numPhotons = round(lessThanN)

                        # Round down apart from initially
                        if numPhotons == 0:
                            numPhotons = 1

                        dict_multiphoton_counts[numPhotons] = dict_multiphoton_counts.get(numPhotons, 0) + 1

                    results_dict["number_of_photons"] += numPhotons

                    if numPhotons == 1:
                        idx_ordered_list = sorted(range(len(islandVals)), key=lambda i: islandVals[i], reverse=True)
                        idx_max = idx_ordered_list[0]
                        ij_tuple = island[idx_max]

                        results_dict["list_countij"].append([1, ij_tuple[0], ij_tuple[1]])
                        continue

                    if self.diagnostic_print:
                        print(f"i ~ {i}, j~ {j}, totVal ~ {totVal}, numPoints = {numPoints}")

                    mat_more_thr, list_count_ij_island_prime,noise_dominated_islands = self.operate_on_mini_island_matrix(
                        island_matrix=matrix_island,
                        num_photons_expected=numPhotons,
                        how_many_more_sigma=self.how_many_more_sigma,
                        noise_dominated_islands=noise_dominated_islands)

                    if diagnostic_of_large_islands:
                        if numPhotons > 7:
                            if unit_test_correct_mat is None:
                                self.plot_matrices_of_island(test_mat=matrix_island,
                                                             island_more_thr=mat_more_thr,
                                                             l_countij_from_algorithm=list_count_ij_island_prime,
                                                             how_many_more_sigma=self.how_many_more_sigma)
                            else:
                                unit_test_correct_mat_island = unit_test_correct_mat[min_i:max_i + 1, min_j:max_j + 1]
                                self.plot_matrices_of_island(test_mat=matrix_island,
                                                             island_more_thr=mat_more_thr,
                                                             l_countij_from_algorithm=list_count_ij_island_prime,
                                                             how_many_more_sigma=self.how_many_more_sigma,
                                                             correctMat=unit_test_correct_mat_island)

                    if list_count_ij_island_prime:
                        # now convert back and store in dictionary
                        for row in list_count_ij_island_prime:
                            results_dict["list_countij"].append([row[0], row[1] + min_i, row[2] + min_j])

        if Report:
            print("-" * 30)
            print("Number of islands: ", results_dict["number_of_islands"])
            print("Number rejected: ", results_dict["number_rejected"])
            print("Number of counts: ", results_dict["number_of_photons"])
            print("Number of ADU counts higher than capture: ", results_dict["number_higher_than_capture"])
            print("Islands with just noise: ", noise_dominated_islands)

            for key in sorted(dict_multiphoton_counts.keys()):
                print(f"There were {dict_multiphoton_counts[key]} events with {key} photons")

        if plot_checkedMat:
            plt.imshow(checked_mat)
            plt.show()

        return results_dict

    def plot_ADU(self, ):

        results_dict = self.operateOnIslands(image_matrix_replace=None)

        list_adu = results_dict["total_ADU"]

        bins_enforced = np.arange(0, 1000, 1)
        hist_values, bin_edges = np.histogram(list_adu, bins=bins_enforced)

        plt.figure(figsize=(10, 5))
        plt.bar(bin_edges[:-1], hist_values, width=1, edgecolor="black", alpha=0.7)
        plt.xlabel("ADU")
        plt.ylabel("Count")
        plt.title(
            f'Histogram of Island ADUs for image {self.index_of_interest} with {self.howManySigma} sigma thresholding')
        plt.grid(True, linestyle="--", alpha=0.5)
        # plt.yscale('log')
        plt.show()

    def operate_on_mini_island_matrix(self, island_matrix, num_photons_expected, how_many_more_sigma=1,noise_dominated_islands=None):
        # Find the position of photons

        if self.diagnostic_print:
            print(f"Given the total ADU we expect {num_photons_expected} photons")

        # further remove ADU:

        islandmat_futher_thresholded = island_matrix.copy()
        islandmat_futher_thresholded = np.where(
            islandmat_futher_thresholded > (self.howManySigma + how_many_more_sigma) * self.sigmaPedestal,
            islandmat_futher_thresholded, 0)

        total_remaining_adu = np.sum(islandmat_futher_thresholded)
        if total_remaining_adu < self.no_p_adu_thr:
            # print(f"Remaining noise at {self.how_many_more_sigma} sigma thresholding")
            if noise_dominated_islands is not None:
                noise_dominated_islands +=1
                return islandmat_futher_thresholded, [], noise_dominated_islands
            else:
                return islandmat_futher_thresholded, []

        list_countij = self.scour_for_neighbours_miniIsland_mat(
            islandmat_futher_thresholded=islandmat_futher_thresholded.copy(),
            num_photons_expected=num_photons_expected,)

        if noise_dominated_islands is not None:
            return islandmat_futher_thresholded, list_countij, noise_dominated_islands
        else:
            return islandmat_futher_thresholded, list_countij


    def scour_for_neighbours_miniIsland_mat(self, islandmat_futher_thresholded, num_photons_expected,):
        nrows, ncols = islandmat_futher_thresholded.shape
        checked_mat_ = np.zeros((nrows, ncols))

        list_count_ij = []
        dict_remaining_islands = {
            "total_ADU": [],
            "island_list": [],
            "adu_of_island_list": [],
        }

        # If an individual point has more than single photon adu thr + offset it feels safe to call it a double hit

        indices_more_than = np.argwhere(islandmat_futher_thresholded > self.sp_adu_thr + self.adu_offset)
        for idx_tuple in indices_more_than:
            val = islandmat_futher_thresholded[idx_tuple[0], idx_tuple[1]]

            if val > 2 * self.sp_adu_thr + self.adu_offset:
                print("3 Photon single pixel event")
                count_single_pixel = 3
            else:
                count_single_pixel = 2

            if count_single_pixel < num_photons_expected + 1:
                list_count_ij.append([count_single_pixel, idx_tuple[0], idx_tuple[1]])
                num_photons_expected -= count_single_pixel

                # set these values to 0 now
                islandmat_futher_thresholded[idx_tuple[0], idx_tuple[1]] = 0
            else:
                print("Fewer Photons expected than that on this individual element")

        def scour_for_neighbours(i_idx, j_idx, island_list, islandValList):
            if i_idx < 0 or i_idx >= nrows or j_idx < 0 or j_idx >= ncols:
                return
            if checked_mat_[i_idx, j_idx] == 1 or islandmat_futher_thresholded[i_idx, j_idx] == 0:
                return
            if len(island_list) > 100:
                return

            checked_mat_[i_idx, j_idx] = 1
            island_list.append((i_idx, j_idx))
            islandValList.append(islandmat_futher_thresholded[i_idx, j_idx])

            # Now scour for neighbours

            for di, dj in [(-1, 0),
                           (0, -1), (0, 1),
                           (1, 0), ]:
                scour_for_neighbours(i_idx + di, j_idx + dj, island_list, islandValList)

        for i in range(nrows):
            for j in range(ncols):
                if islandmat_futher_thresholded[i, j] != 0 and checked_mat_[i, j] == 0:
                    island = []
                    island_vals = []
                    scour_for_neighbours(i, j, island, island_vals)

                    totval = sum(island_vals)

                    if self.diagnostic_print:
                        print(totval)

                    dict_remaining_islands["total_ADU"].append(totval)
                    dict_remaining_islands["island_list"].append(island)
                    dict_remaining_islands["adu_of_island_list"].append(island_vals)

        proportion_of_total = self.split_integer_adu(num_photons_expected, dict_remaining_islands,
                                                     matrix_isl=islandmat_futher_thresholded)

        # Now split their quantities between the points in the island

        if self.diagnostic_print:
            print("The number of splittings is: ", proportion_of_total)

        for numPhotons_island, ij_list, adu_list in zip(proportion_of_total, dict_remaining_islands["island_list"],
                                                        dict_remaining_islands["adu_of_island_list"]):

            idx_ordered_list = sorted(range(len(adu_list)), key=lambda i: adu_list[i], reverse=True)

            if len(ij_list) < numPhotons_island:
                idx_max = idx_ordered_list[0]
                ij_tuple = ij_list[idx_max]
                list_count_ij.append([numPhotons_island, ij_tuple[0], ij_tuple[1]])
                if self.diagnostic_print:
                    print([numPhotons_island, ij_tuple[0], ij_tuple[1]])
            else:
                for number in range(numPhotons_island):
                    idx_photon = idx_ordered_list[number]  # ie find the idx of this photon
                    ij_tuple = ij_list[idx_photon]

                    list_count_ij.append([1, ij_tuple[0], ij_tuple[1]])

                    if self.diagnostic_print:
                        print([1, ij_tuple[0], ij_tuple[1]])

        if self.diagnostic_print:
            print("-" * 30)
        return list_count_ij

    @staticmethod
    def split_integer_adu(num_photons_expected, dict_remaining_islands, matrix_isl=None):
        remaining_adu_tot = sum(dict_remaining_islands["total_ADU"])

        # Compute the ideal split values as floats
        split_floats = [(q / remaining_adu_tot) * num_photons_expected for q in dict_remaining_islands["total_ADU"]]

        # Get initial integer splits by flooring values
        split_values = [int(x) for x in split_floats]

        # Compute remaining amount to distribute
        remainder = int(round(num_photons_expected - sum(split_values)))

        # Distribute remainder to the indices with the highest decimal parts
        decimal_parts = [(i, split_floats[i] - split_values[i]) for i in
                         range(len(dict_remaining_islands["total_ADU"]))]
        decimal_parts.sort(key=lambda x: x[1], reverse=True)

        try:
            for i in range(remainder):
                split_values[decimal_parts[i][0]] += 1
        except IndexError:
            print("Index out of range")
            if matrix_isl is not None:
                plt.imshow(matrix_isl)
                plt.show()

        return split_values

    @staticmethod
    def plot_matrices_of_island(test_mat, island_more_thr, l_countij_from_algorithm, how_many_more_sigma,
                                correctMat=None):
        matrix_found = np.zeros(test_mat.shape)
        for row in l_countij_from_algorithm:
            matrix_found[row[1], row[2]] = row[0]

        total_adu = np.sum(test_mat)

        if correctMat is None:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1), plt.imshow(test_mat, cmap='hot'), plt.title(f"Unit test matrix\n{total_adu} ADU")
            plt.subplot(1, 3, 2), plt.imshow(island_more_thr, cmap='hot'), plt.title(
                f"Unit test matrix\n+{how_many_more_sigma} sigma thresholding")
            plt.subplot(1, 3, 3), plt.imshow(matrix_found, cmap='hot'), plt.title("Island Counted Photons")
            plt.show()
        else:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 4, 1), plt.imshow(test_mat, cmap='hot'), plt.title(f"Unit test matrix\n{total_adu} ADU")
            plt.subplot(1, 4, 2), plt.imshow(island_more_thr, cmap='hot'), plt.title(
                f"Unit test matrix\n+{how_many_more_sigma} sigma thresholding")
            plt.subplot(1, 4, 3), plt.imshow(matrix_found, cmap='hot'), plt.title("Island Counted Photons")
            plt.subplot(1, 4, 4), plt.imshow(correctMat, cmap='hot'), plt.title("Unit Test Photons")
            plt.show()



class Island_PC_with_presetShape:
    def __init__(self, indexOfInterest,
                 no_photon_adu_thr=100,
                 sp_adu_cutoff=225,
                 two_photon_cutoff=325,
                 howManySigma_thr=1,
                 removeRows0To_=0
                 ):

        def printVar():
            print("-" * 30)
            print("class PhotonCounting initiated with:")
            print("Index of Interest: ", indexOfInterest)
            print("No Photon ADU Threshold: ", no_photon_adu_thr, "; Single Photon ADU Threshold: ", sp_adu_cutoff)
            print("Two Photon ADU cutoff: ", two_photon_cutoff)
            if removeRows0To_ > 0:
                print("Remove Rows 0 to: ", removeRows0To_)
            print(f"imMat has the mean removed and is then thresholded above {howManySigma_thr} sigma ")

        printVar()

        self.imMatRAW = loadData()[indexOfInterest]
        self.index_of_interest = indexOfInterest

        if removeRows0To_ > 0:
            self.imMatRAW = self.imMatRAW[removeRows0To_:, :]

        self.no_p_adu_thr = no_photon_adu_thr
        self.sp_adu_cutoff = sp_adu_cutoff
        self.two_photon_cutoff = two_photon_cutoff

        meanPedestal, sigmaPedestal = pedestal_mean_sigma_awayFromLines(self.imMatRAW, indexOfInterest=indexOfInterest)
        self.meanPedestal = meanPedestal
        self.sigmaPedestal = sigmaPedestal

        def removeMeanPedestal():
            mat_minusMean = self.imMatRAW.astype(np.int16) - self.meanPedestal
            mat_minusMean[mat_minusMean < 0] = 0
            return mat_minusMean

        self.imMatMeanRemoved = removeMeanPedestal()
        # Calling this self.imMat as well due to old version
        self.howManySigma = howManySigma_thr
        self.imMat = np.where(self.imMatMeanRemoved > howManySigma_thr * self.sigmaPedestal, self.imMatMeanRemoved, 0)

    def operateOnIslands(self, image_matrix_replace=None, plot_checkedMat=False, diagnosticPrint=True):
        print("-" * 30)
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
                    elif totVal < self.sp_adu_cutoff:
                        numPhotons = 1
                    elif totVal < self.two_photon_cutoff:
                        numPhotons = 2
                    else:
                        if diagnosticPrint:
                            print(f"There was more than the cap i~{i}, j~{j}, totVal = {totVal}")
                        results_dict["number_higher_than_capture"] += 1
                        continue

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


class Preset_Shape_PhotonCounting:
    def __init__(self, index_of_interest, how_many_sigma=1, folderpath="stored_variables"):
        self.indexOI = index_of_interest
        self.howManySigma = how_many_sigma
        self.folderpath = folderpath

        self.imMatRaw = imData[index_of_interest]

        mat_minusMean, thr = mat_minusMean_thr_aboveNsigma(index_of_interest, how_many_sigma)

        self.imMat = mat_minusMean

        self.unique_keys = self.kernel_Dict().keys()

    def list_aduCount_i_j(self, save=True, diagnosticPrint=False, plot_ADU_hist=False):
        print("-" * 30)
        print("list_aduCount_i_j")

        startTime = time.time()

        dict_shapes = self.shape_KernelDict()
        unique_keys = self.kernel_Dict().keys()

        if diagnosticPrint:
            print("\nDiagnostic Print:")
            print(f"dict_shapes_keys = {dict_shapes.keys()}")
            print(f"unique_keys = {unique_keys}")

        output_dictionaries = {}
        for key in unique_keys:
            # Initialise Counts
            count_found = 0
            list_adu_ij = []

            output_dictionaries[key] = {
                "count_found": count_found,
                "list_adu_ij": list_adu_ij,
            }

        imMat = self.imMat.copy()
        imMatRaw_points_removed = self.imMatRaw.copy()

        rowNum, colNum = imMat.shape
        im_binary = np.where(imMat > 0, 1, 0)

        checked_mat = np.zeros_like(imMat)

        for shape_tuple in dict_shapes.keys():
            print("shape_tuple = ", shape_tuple)
            dict_kernel_type = dict_shapes[shape_tuple]
            k_rows, k_cols = shape_tuple
            for i in range(rowNum - k_rows + 1):
                for j in range(colNum - k_cols + 1):
                    # Consider areas of the same size as the kernel:

                    # Note this only is needed to account for the l with i+1 and j+1 not being a part of the shape
                    if shape_tuple == (4, 4):
                        if checked_mat[i + 1, j + 1] and checked_mat[i + 2, j + 2] == 1:
                            continue
                    else:
                        if checked_mat[i + 1, j + 1] == 1:
                            continue

                    convolvedArea = im_binary[i:i + k_rows, j:j + k_cols]

                    if np.all(convolvedArea == 0):
                        checked_mat[i:i + k_rows, j:j + k_cols] = 1

                    for kernel_type in dict_kernel_type.keys():
                        kdict = dict_kernel_type[kernel_type]
                        kernels = kdict["kernels"]
                        masks = kdict["masks"]
                        nonZeroIndicesKernels = kdict["nonZero_idx_kernel"]
                        nonZeroIndicesMasks = kdict["nonZero_idx_mask"]

                        for kernel, mask, idxKernel, idxMask in zip(kernels, masks, nonZeroIndicesKernels,
                                                                    nonZeroIndicesMasks):
                            if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel,
                                                                                             mask)):
                                continue

                            outputDict_kt = output_dictionaries[kernel_type]
                            outputDict_kt["count_found"] += 1

                            # Finding how many points there are in the kernel that fits
                            # and finding their indices
                            numPoints_kernel = len(idxKernel)
                            dict_idx = {}
                            for point_number in range(numPoints_kernel):
                                dict_idx[point_number] = idxKernel[point_number]

                            for idx in idxKernel:
                                checked_mat[i + idx[0], j + idx[1]] = 1
                            for idx in idxMask:
                                checked_mat[i + idx[0], j + idx[1]] = 1

                            totVal = 0

                            dict_vals = {}
                            for key in dict_idx.keys():
                                val_key = self.imMat[i + dict_idx[key][0], j + dict_idx[key][1]]
                                dict_vals[key] = val_key
                                totVal += val_key

                                imMatRaw_points_removed[i + dict_idx[key][0], j + dict_idx[key][1]] = 0

                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            max_key = keyOrderedList[0]
                            outputDict_kt["list_adu_ij"].append(
                                [totVal, i + dict_idx[max_key][0], j + + dict_idx[max_key][0]])

        endtime = time.time()
        function_time = endtime - startTime
        minutes, seconds = divmod(function_time, 60)
        print(f"list_aduCount_i_j function runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

        adu_dict = {}
        if plot_ADU_hist:
            list_adu_ij_combined = []
            for key in unique_keys:
                list_adu_ij_key = output_dictionaries[key]["list_adu_ij"]
                list_adu_ij_combined.extend(list_adu_ij_key)
                arr_adu_ij_key = np.array(list_adu_ij_key)

                # Extract only ADU column (first column)
                adu_values = arr_adu_ij_key[:, 0]

                # Store in dictionary with the key as label
                adu_dict[key] = adu_values
        else:
            list_adu_ij_combined = []
            for key in unique_keys:
                list_adu_ij_key = output_dictionaries[key]["list_adu_ij"]
                list_adu_ij_combined.extend(list_adu_ij_key)

        if save:
            spc_folderpath = spc_folder(self.folderpath, self.indexOI)
            sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")
            if not os.path.exists(sigma_folderpath):
                os.makedirs(sigma_folderpath)

            for key in unique_keys:
                filepath = os.path.join(sigma_folderpath, f"{key}_adu_i_j.npy")
                list_adu_ij_key = output_dictionaries[key]["list_adu_ij"]
                arr_adu_ij_key = np.array(list_adu_ij_key)

                np.save(filepath, arr_adu_ij_key)

            matrix_filepath = os.path.join(sigma_folderpath, f"raw_matrix_with_checked_points_removed.npy")
            np.save(matrix_filepath, imMatRaw_points_removed)

        if plot_ADU_hist:
            bins_enforced = np.arange(0, 1000, 1)

            # Plot stacked histogram
            plt.figure(figsize=(10, 5))
            plt.hist(adu_dict.values(), bins=bins_enforced, stacked=True, label=adu_dict.keys(), alpha=0.7,
                     edgecolor="black")

            plt.xlabel("ADU")
            plt.ylabel("Frequency")
            plt.title(f"Stacked Histogram of ADU Values with {self.howManySigma} sigma thresholding")
            plt.legend(title="Keys")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

        if diagnosticPrint:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(imMat[600:701, 1400:1500], cmap='hot'), plt.title(f"imMat")
            plt.subplot(1, 2, 2), plt.imshow(imMatRaw_points_removed[600:701, 1400:1500], cmap='hot'), plt.title(
                f"imMat_points_removed")
            plt.show()

        return output_dictionaries, imMatRaw_points_removed

    def shape_KernelDict(self):
        kernel_dict = self.kernel_Dict()
        dictionary_shapes = {}

        for key in kernel_dict.keys():
            dict_of_key = kernel_dict[key]
            kernelList = dict_of_key["kernels"]
            maskList = dict_of_key["masks"]

            for kernel, mask in zip(kernelList, maskList):

                shape_kernel = kernel.shape

                if shape_kernel not in dictionary_shapes:
                    dictionary_shapes[shape_kernel] = {}  # Initialize an empty dictionary

                # Initialising position of non-zero points here for easier capture later

                if key not in dictionary_shapes[shape_kernel]:
                    dictionary_shapes[shape_kernel][key] = {
                        "kernels": [kernel],
                        "masks": [mask],
                        "nonZero_idx_kernel": [np.argwhere(kernel != 0)],
                        "nonZero_idx_mask": [np.argwhere(mask != 0)],

                    }
                else:
                    # Append kernel and mask to the existing lists
                    dictionary_shapes[shape_kernel][key]["kernels"].append(kernel)
                    dictionary_shapes[shape_kernel][key]["masks"].append(mask)
                    dictionary_shapes[shape_kernel][key]["nonZero_idx_kernel"].append(np.argwhere(kernel != 0))
                    dictionary_shapes[shape_kernel][key]["nonZero_idx_mask"].append(np.argwhere(mask != 0))

        return dictionary_shapes

    def preset_then_island(self, plot_ADU_hist=False, save=True, island_thresholding_sigma=None):

        if island_thresholding_sigma is None:
            island_thresholding_sigma = self.howManySigma


        island_eng = Island_PhotonCounting(indexOfInterest=self.indexOI)
        ped_mean = island_eng.meanPedestal
        ped_sigma = island_eng.sigmaPedestal

        try:
            spc_folderpath = spc_folder(self.folderpath, self.indexOI)
            sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")

            adu_dict = {}

            for key in self.kernel_Dict().keys():
                filepath = os.path.join(sigma_folderpath, f"{key}_adu_i_j.npy")
                arr_adu_ij_key = np.load(filepath)
                adu_values = arr_adu_ij_key[:, 0]
                adu_dict[key] = adu_values

            matrix_filepath = os.path.join(sigma_folderpath, f"raw_matrix_with_checked_points_removed.npy")
            imMatRaw_points_removed = np.load(matrix_filepath)

            adu_island_filename = "island_adu_list.npy"

            arr_adu_island = np.load(os.path.join(sigma_folderpath, adu_island_filename))

            adu_dict["Island"] = np.array(arr_adu_island)

        except FileNotFoundError as e:
            print(f"File not found: {e}")
            print("Running list_aduCount_i_j")
            output_dict_list_aduCount_i_j, imMatRaw_points_removed = self.list_aduCount_i_j()

            adu_dict = {}
            for key in output_dict_list_aduCount_i_j.keys():
                list_adu_ij_key = output_dict_list_aduCount_i_j[key]["list_adu_ij"]
                arr_adu_ij_key = np.array(list_adu_ij_key)
                # Extract only ADU column (first column)
                adu_values = arr_adu_ij_key[:, 0]
                # Store in dictionary with the key as label
                adu_dict[key] = adu_values

            # removing left edge effects
            imMatRaw_points_removed[:, 0:3] = 0
            imMat_p_removed_islandsigma_thresholded = mat_min_mean_thr_above_Nsigma2(matrix=imMatRaw_points_removed,
                                                                                     mean=ped_mean, sigma=ped_sigma,
                                                                                     n_sigma=island_thresholding_sigma)
            results_dict_island = island_eng.operateOnIslands(image_matrix_replace=imMat_p_removed_islandsigma_thresholded,)
            list_adu_island = results_dict_island["total_ADU"]
            list_count_ij_island = results_dict_island["list_countij"]

            if list_count_ij_island is None:
                print("OH NO")
                raise ValueError

            if save:
                spc_folderpath = spc_folder(self.folderpath, self.indexOI)
                sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")
                adu_island_filename = "island_adu_list.npy"
                countij_island_filename = "island_countij.npy"

                np.save(os.path.join(sigma_folderpath, adu_island_filename), np.array(list_adu_island))
                np.save(os.path.join(sigma_folderpath, countij_island_filename), np.array(list_count_ij_island))

            adu_dict["Island"] = np.array(list_adu_island)

        if plot_ADU_hist:
            bins_enforced = np.arange(0, 1000, 1)

            # Plot stacked histogram
            plt.figure(figsize=(10, 5))
            plt.hist(adu_dict.values(), bins=bins_enforced, stacked=True, label=adu_dict.keys(), alpha=0.7,
                     edgecolor="black")

            plt.xlabel("ADU")
            plt.ylabel("Count")
            plt.title(
                f"Image {self.indexOI}: Stacked Histogram of ADU Values with thresholding:\n{self.howManySigma} sigma for preset shapes\n{island_thresholding_sigma} sigma for islanding")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

    def find_just_island(self, save=True, island_thresholding_sigma=None):

        if island_thresholding_sigma is None:
            island_thresholding_sigma = self.howManySigma

        island_eng = Island_PhotonCounting(indexOfInterest=self.indexOI)
        ped_mean = island_eng.meanPedestal
        ped_sigma = island_eng.sigmaPedestal

        try:
            spc_folderpath = spc_folder(self.folderpath, self.indexOI)
            sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")

            matrix_filepath = os.path.join(sigma_folderpath, f"raw_matrix_with_checked_points_removed.npy")
            imMatRaw_points_removed = np.load(matrix_filepath)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            print("Running list_aduCount_i_j")
            output_dict_list_aduCount_i_j, imMatRaw_points_removed = self.list_aduCount_i_j()

        # removing left edge effects
        imMatRaw_points_removed[:, 0:3] = 0
        imMat_p_removed_islandsigma_thresholded = mat_min_mean_thr_above_Nsigma2(matrix=imMatRaw_points_removed,
                                                                                 mean=ped_mean, sigma=ped_sigma,
                                                                                 n_sigma=island_thresholding_sigma)
        results_dict_island = island_eng.operateOnIslands(image_matrix_replace=imMat_p_removed_islandsigma_thresholded,)
        list_adu_island = results_dict_island["total_ADU"]
        list_count_ij_island = results_dict_island["list_countij"]

        # print("list_count_ij_island: ",list_count_ij_island)

        if save:
            spc_folderpath = spc_folder(self.folderpath, self.indexOI)
            sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")
            adu_island_filename = "island_adu_list.npy"
            countij_island_filename = "island_countij.npy"

            np.save(os.path.join(sigma_folderpath, adu_island_filename), np.array(list_adu_island))
            np.save(os.path.join(sigma_folderpath, countij_island_filename), np.array(list_count_ij_island))

        return list_count_ij_island

    @staticmethod
    def kernel_Dict():
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

        def double_pixel():
            dp_isolated_kernel1 = np.array([[0, 0, 0, 0],
                                            [0, 1, 1, 0],
                                            [0, 0, 0, 0]])
            dp_check_mask1 = np.array([[0, 1, 1, 0],
                                       [1, 0, 0, 1],
                                       [0, 1, 1, 0]])

            dp_isolated_kernel2 = np.rot90(dp_isolated_kernel1)
            dp_check_mask2 = np.rot90(dp_check_mask1)

            return {
                "kernels": [dp_isolated_kernel1, dp_isolated_kernel2],
                "masks": [dp_check_mask1, dp_check_mask2],
            }

        def triple_pixel():
            tp_kernel_1 = np.array([[0, 0, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0]])
            tp_kernel_2 = np.rot90(tp_kernel_1)
            tp_kernel_3 = np.rot90(tp_kernel_2)
            tp_kernel_4 = np.rot90(tp_kernel_3)

            tp_mask_1 = np.array([[0, 1, 1, 0],
                                  [1, 0, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0]])
            tp_mask_2 = np.rot90(tp_mask_1)
            tp_mask_3 = np.rot90(tp_mask_2)
            tp_mask_4 = np.rot90(tp_mask_3)

            return {
                "kernels": [tp_kernel_1, tp_kernel_2, tp_kernel_3, tp_kernel_4],
                "masks": [tp_mask_1, tp_mask_2, tp_mask_3, tp_mask_4],
            }

        def quadruple_pixel():
            qp_kernel_1 = np.array([[0, 0, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 0, 0]])
            qp_mask_1 = np.array([[0, 1, 1, 0],
                                  [1, 0, 0, 1],
                                  [1, 0, 0, 1],
                                  [0, 1, 1, 0]])

            return {
                "kernels": [qp_kernel_1],
                "masks": [qp_mask_1],
            }

        return {
            "single_pixel": single_pixel(),
            "double_pixel": double_pixel(),
            "triple_pixel": triple_pixel(),
            "quadruple_pixel": quadruple_pixel(),
        }

    def access_saved_adu_ij(self, return_raw_mat_points_removed=False):

        dict_arrays = {}
        try:
            spc_folderpath = spc_folder(self.folderpath, self.indexOI)
            sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")

            for key in self.kernel_Dict().keys():
                filepath = os.path.join(sigma_folderpath, f"{key}_adu_i_j.npy")
                arr_adu_ij_key = np.load(filepath)

                dict_arrays[key] = arr_adu_ij_key

            matrix_filepath = os.path.join(sigma_folderpath, f"raw_matrix_with_checked_points_removed.npy")
            imMatRaw_points_removed = np.load(matrix_filepath)

        except FileNotFoundError as e:
            print(f"File not found: {e}")
            print("Running list_aduCount_i_j")
            output_dict_list_aduCount_i_j, imMatRaw_points_removed = self.list_aduCount_i_j()

            for key in output_dict_list_aduCount_i_j.keys():
                list_adu_ij_key = output_dict_list_aduCount_i_j[key]["list_adu_ij"]
                arr_adu_ij_key = np.array(list_adu_ij_key)

                dict_arrays[key] = arr_adu_ij_key
        if return_raw_mat_points_removed:
            return dict_arrays, imMatRaw_points_removed
        else:
            return dict_arrays

    def access_saved_island_countij(self):

        try:
            spc_folderpath = spc_folder(self.folderpath, self.indexOI)
            sigma_folderpath = os.path.join(spc_folderpath, f"{self.howManySigma}_sigma")
            filename = "island_countij.npy"

            arr_count_ij = np.load(os.path.join(sigma_folderpath, filename))

            return arr_count_ij

        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return np.array(self.find_just_island())


class Unit_testing:
    def __init__(self,index_of_interest=None):
        self.index_of_interest = index_of_interest
        self.mat_minus_Mean = matMinusMean(self.index_of_interest)


    @staticmethod
    def compare_islandPhotons_unit_test_hits(fillFrac=0.1, matrix_size=(2048, 2048), mean_adu=150, std_adu=10,
                                             seed=125, no_photon_adu_thr=100, sp_adu_thr=150, adu_offset=30,
                                             adu_cap=3000, diagnostics=True, diagnostic_of_large_islands=False):
        print("compare_islandPhotons_unit_test_hits\n")
        spcTrain = SPC_Train_images(2)

        unit_test_mat, photon_hits_mat = spcTrain.createTestData(fillfraction=fillFrac, matrix_size=matrix_size,
                                                                 mean_adu=mean_adu,
                                                                 std_adu=std_adu, seed=seed,
                                                                 addAnomaly=False, return_mat_with_exact_hits=True,
                                                                 )

        num_photons = fillFrac * matrix_size[0] * matrix_size[1]
        print("The number of Photons created was: ", num_photons)

        pc_eng = Island_PhotonCounting(8, no_photon_adu_thr=no_photon_adu_thr, sp_adu_thr=sp_adu_thr,
                                       adu_offset=adu_offset,
                                       adu_cap=adu_cap,
                                       removeRows0To_=0, howManySigma_thr=2, )

        results_dict = pc_eng.operateOnIslands(image_matrix_replace=unit_test_mat,
                                               diagnostic_of_large_islands=diagnostic_of_large_islands,
                                               unit_test_correct_mat=photon_hits_mat,)

        mat_islandPhotons = np.zeros(matrix_size)
        list_count_ij = results_dict["list_countij"]
        for row in list_count_ij:
            mat_islandPhotons[row[1], row[2]] = row[0]

        mat_difference = photon_hits_mat - mat_islandPhotons

        # success metric: how many photons are incorrectly

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1), plt.imshow(unit_test_mat, cmap='hot'), plt.title("Unit test matrix")
        plt.subplot(1, 4, 2), plt.imshow(photon_hits_mat, cmap='hot'), plt.title("Unit test photon hit matrix")
        plt.subplot(1, 4, 3), plt.imshow(mat_islandPhotons, cmap='hot'), plt.title("Island Counted Photons")
        plt.subplot(1, 4, 4), plt.imshow(mat_difference, cmap='hot'), plt.title("Difference")
        plt.show()


    def find_approximate_fill_fraction(self, threshold=60, sp_adu=150):
        mat_minus_mean = self.mat_minus_Mean

        nrows = mat_minus_mean.shape[0]
        ncols = mat_minus_mean.shape[1]

        mat_above_thr = np.where(mat_minus_mean > threshold, mat_minus_mean, 0)

        total_adu = np.sum(mat_above_thr)

        approx_N_photons = total_adu/sp_adu
        n_pixels = nrows * ncols

        fill_frac = approx_N_photons / n_pixels

        return fill_frac


    def find_uncertainty_in_spc_engine(self, diagnostics=False, unit_test_mat_size=(1000,1000)):
        if self.index_of_interest is None:
            raise ValueError("index_of_interest cannot be None")

        fill_frac = self.find_approximate_fill_fraction()

        if diagnostics:
            print("\nPerforming Unit Test:")
            print("The fill fraction is: ", fill_frac)

        spcTrain = SPC_Train_images(how_many_sigma=2,indexOfInterest=self.index_of_interest,)
        unit_test_mat, photon_hits_mat = spcTrain.createTestData(fillfraction=fill_frac, matrix_size=unit_test_mat_size,
                                                                 mean_adu=150,
                                                                 std_adu=20,return_mat_with_exact_hits=True)

        pc_eng = Island_PhotonCounting(self.index_of_interest,diagnostic_print=False,declareVars=False)
        results_dict = pc_eng.operateOnIslands(image_matrix_replace=unit_test_mat,unit_test_correct_mat=photon_hits_mat,Report=False )

        mat_islandPhotons = np.zeros(unit_test_mat_size)
        list_count_ij = results_dict["list_countij"]
        for row in list_count_ij:
            mat_islandPhotons[row[1], row[2]] = row[0]

        mat_difference = photon_hits_mat - mat_islandPhotons

        # success metric: how many photons are incorrectly

        # If pixels are 1 away from one another then they cancel out. Otherwise collect
        remaining_positives, remaining_negatives = self.cancel_adjacent_values_and_collect_remaining_elements(mat_difference)

        if diagnostics:
            print(f"The number of photons not captured and without nearby elements: ", remaining_positives)
            print(f"The number of photons the model guesses that aren't real: ", remaining_negatives)
        # The success Standard is based off the rate of guesses that are wrong

        numPhotons = fill_frac*unit_test_mat_size[0]*unit_test_mat_size[1]

        error_fraction = remaining_negatives / numPhotons
        lost_fraction = remaining_positives/numPhotons

        remaining_positives_without_cancellation = np.sum(mat_difference == 1)
        remaining_negatives_without_cancellation = np.sum(mat_difference == -1)
        ef_without_cancel = remaining_positives_without_cancellation/numPhotons
        lf_without_cancel = remaining_negatives_without_cancellation/numPhotons

        if diagnostics:
            print("Total Photons = ", numPhotons)
            print(f"error_fraction is {error_fraction}")
            print(f"loss_fraction is {lost_fraction}")

            print("without adjacent cancellation:")
            print(f"error_fraction is {ef_without_cancel}")
            print(f"loss_fraction is {lf_without_cancel}")

        if diagnostics:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 4, 1), plt.imshow(unit_test_mat, cmap='hot'), plt.title("Unit test matrix")
            plt.subplot(1, 4, 2), plt.imshow(photon_hits_mat, cmap='hot'), plt.title("Unit test photon hit matrix")
            plt.subplot(1, 4, 3), plt.imshow(mat_islandPhotons, cmap='hot'), plt.title("Island Counted Photons")
            plt.subplot(1, 4, 4), plt.imshow(mat_difference, cmap='hot'), plt.title(f"Difference: Error of {error_fraction*100}%")
            plt.show()

        return error_fraction + lost_fraction


    @staticmethod
    def cancel_adjacent_values_and_collect_remaining_elements(matrix_OI):
        rows, cols = matrix_OI.shape
        changes = True

        # Define 8-neighbor directions (adjacent and diagonal)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Top row
            (0, -1), (0, 1),  # Sides
            (1, -1), (1, 0), (1, 1)  # Bottom row
        ]

        while changes:
            changes = False
            new_matrix = matrix_OI.copy()

            for i in range(rows):
                for j in range(cols):
                    if matrix_OI[i, j] == 1 or matrix_OI[i, j] == -1:
                        for dr, dc in directions:
                            nr, nc = i + dr, j + dc
                            if 0 <= nr < rows and 0 <= nc < cols and matrix_OI[nr, nc] == -matrix_OI[i, j]:
                                # Cancel out
                                new_matrix[i, j] = 0
                                new_matrix[nr, nc] = 0
                                changes = True
                                break

            matrix_OI = new_matrix

        # Count remaining +1s and -1s
        remaining_positives = np.sum(matrix_OI == 1)
        remaining_negatives = np.sum(matrix_OI == -1)

        return remaining_positives, remaining_negatives

    @staticmethod
    def unitTest1(fillFrac=0.1, matrix_size=(2048, 2048), mean_adu=150, std_adu=10,
                  seed=125, plotMat=False, ):
        spcTrain = SPC_Train_images(2)

        unit_test_mat, photon_hits_mat = spcTrain.createTestData(fillfraction=fillFrac, matrix_size=matrix_size,
                                                                 mean_adu=mean_adu,
                                                                 std_adu=std_adu, seed=seed,
                                                                 addAnomaly=False, return_mat_with_exact_hits=True)

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
            pc_eng = Island_PhotonCounting(8, no_photon_adu_thr=params[0], sp_adu_thr=params[1], adu_offset=params[2],
                                           adu_cap=1650,
                                           removeRows0To_=0, howManySigma_thr=2, )

            results_dict = pc_eng.operateOnIslands(image_matrix_replace=unit_test_mat)

            numCaptured = results_dict["number_of_photons"]

            loss = (num_photons - numCaptured) ** 2

            return loss

        result = minimize(loss_function, adu_thr_guess, method='Nelder-Mead', options={'maxiter': 30})

        print(result.x)

    @staticmethod
    def test_im8clusters():
        mat_cluster_1 = TestImages().image8_cluster_reducedScheme1()
        mat_cluster_2 = TestImages().image8_cluster_reducedScheme2()
        mat_cluster_3 = TestImages().image8_cluster_reducedScheme3()

        pc_eng = Island_PhotonCounting(8,
                                       removeRows0To_=0, howManySigma_thr=2, )

        print("Cluster1")
        pc_eng.operateOnIslands(image_matrix_replace=mat_cluster_1, )
        print("Cluster2")
        pc_eng.operateOnIslands(image_matrix_replace=mat_cluster_2, )
        print("Cluster3")
        pc_eng.operateOnIslands(image_matrix_replace=mat_cluster_3, )

        image8ClusterShow()


def access_unc_spc_eng(index_of_interest,folderpath="stored_variables"):
    index_folder = os.path.join(folderpath,str(index_of_interest))




if __name__ == "__main__":

    def test_OperateOnIslands(indexOI):
        pc_eng = Island_PhotonCounting(indexOI,)
        pc_eng.operateOnIslands()


    # test_OperateOnIslands(8)
    # test_OperateOnIslands(11)

    # Unit_testing(11).find_uncertainty_in_spc_engine(True)

    def investigate_totADU(indexOI, how_many_sigma=1):
        print("Investigating TOTAL ADU")
        pc_eng = Island_PhotonCounting(indexOI, howManySigma_thr=how_many_sigma, )

        plt.imshow(pc_eng.imMat)
        plt.title(f"Image {indexOI} with {how_many_sigma} sigma thresholding")
        plt.show()

        pc_eng.plot_ADU()


    # investigate_totADU(11,3)

    def presetShape_PC(indexOI_, howManySig=2, island_thresholding_sigma=2):
        preset_eng = Preset_Shape_PhotonCounting(indexOI_, howManySig)
        # preset_eng.list_aduCount_i_j(diagnosticPrint=True,save=True,plot_ADU_hist=True)
        preset_eng.preset_then_island(True, island_thresholding_sigma=island_thresholding_sigma)

        # preset_eng.find_just_island(save=True)


    # presetShape_PC(11)
    # presetShape_PC(8)

    # plot_compare_singlePixel_ADU(ylim=1000)

    # save_SinglePixel_ADU(plot=True,how_many_sigma=1,ylim=1000,save=False)

    # Unit_testing().compare_islandPhotons_unit_test_hits(fillFrac=0.1, matrix_size=(100, 100), diagnostics=True,diagnostic_of_large_islands=True)


    def test_mini_island_method(how_many_more_sigma=1):
        island_eng__ = Island_PhotonCounting(8, )

        test_mat = TestImages().image8_cluster_reduced_edited_moreDifficult()

        island_more_thr, l_countij = island_eng__.operate_on_mini_island_matrix(island_matrix=test_mat,
                                                                                num_photons_expected=8,
                                                                                how_many_more_sigma=how_many_more_sigma,
                                                                                diagnostic_print=True)

        matrix_found = np.zeros(test_mat.shape)
        for row in l_countij:
            matrix_found[row[1], row[2]] = row[0]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1), plt.imshow(test_mat, cmap='hot'), plt.title("Unit test matrix")
        plt.subplot(1, 3, 2), plt.imshow(island_more_thr, cmap='hot'), plt.title(
            f"Unit test matrix\n+{how_many_more_sigma} sigma thresholding")
        plt.subplot(1, 3, 3), plt.imshow(matrix_found, cmap='hot'), plt.title("Island Counted Photons")
        plt.show()


    # test_mini_island_method()

    # Unit_testing().test_im8clusters()

    def CheckAllSinglePixelHist(indices_of_interest=list_good_data):
        print("Checking all single-pixel histograms")
        for index_ in indices_of_interest:
            SinglePixel(index_, 2).plot_adu_dist(bins=200)


    # CheckAllSinglePixelHist([16])

    def compareFaveLittleSpot():

        pc = Island_PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=2)
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
