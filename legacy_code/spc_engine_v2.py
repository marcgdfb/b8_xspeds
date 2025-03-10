import os
from pedestal_engine_v2 import *
from tools import *
from testImages import *

im8_filepath = r"/data_logs/image_matrices/image_8"

# line of 3
# shape of 5
# produce an algorithm that will search with shapes and save array (in reduced scheme) in data logs

def kernelDict():
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

    def quintuple_pixel():
        quint_kernel_1 = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
        quint_kernel_2 = np.rot90(quint_kernel_1)
        quint_kernel_3 = np.rot90(quint_kernel_2)
        quint_kernel_4 = np.rot90(quint_kernel_3)

        quint_mask_1 = np.array([[0, 1, 1, 1, 0],
                              [1, 0, 0, 0, 1],
                              [0, 1, 0, 0, 1],
                              [0, 0, 1, 1, 0]])
        quint_mask_2 = np.rot90(quint_mask_1)
        quint_mask_3 = np.rot90(quint_mask_2)
        quint_mask_4 = np.rot90(quint_mask_3)

        return {
            "kernels": [quint_kernel_1, quint_kernel_2, quint_kernel_3, quint_kernel_4],
            "masks": [quint_mask_1, quint_mask_2, quint_mask_3, quint_mask_4],
        }

    def tp_line():
        tp_line_kernel_1 = np.array([[0, 0, 0, 0,0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0]])
        tp_line_kernel_2 = np.rot90(tp_line_kernel_1)

        tp_line_mask_1 = np.array([[0, 1, 1, 1, 0],
                                   [1, 0, 0, 0, 1],
                                   [0, 1, 1, 1, 0]])
        tp_line_mask_2 = np.rot90(tp_line_mask_1)

        return {
            "kernels": [tp_line_kernel_1, tp_line_kernel_2],
            "masks": [tp_line_mask_1, tp_line_mask_2],
        }

    return {
        "single_pixel": single_pixel(),
        "double_pixel": double_pixel(),
        "triple_pixel": triple_pixel(),
        "quadruple_pixel": quadruple_pixel(),
        "quintuple_pixel": quintuple_pixel(),
        "tp_line": tp_line(),
    }


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


kdic = kernelDict()


class PhotonCounting:
    def __init__(self, indexOfInterest, no_photon_adu_thr=50, sp_adu_thr=180, dp_adu_thr=240,
                 removeRows0To_=0, howManySigma_thr=2,):

        def printVar():
            print("-" * 30)
            print("class PhotonCounting initiated with:")
            print("Index of Interest: ", indexOfInterest)
            print("No Photon ADU Threshold: ", no_photon_adu_thr, "; Single Photon ADU Threshold: ", sp_adu_thr,
                  "; Double Photon ADU Threshold: ", dp_adu_thr)

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
        self.dp_adu_thr = dp_adu_thr

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

    def check_kernel_type(self, kernel_type,diagnostics=False,
                          report=False, image_matrix_replace=None):
        # Initialise Counts
        count_found = 0
        countReject = 0
        count_1photon = 0
        count_2photon = 0
        count_morethan2 = 0

        countPerfect = 0
        count_withDiagonal = 0

        list_countij = []
        list_ADU_sum = []

        outputDict_initialised = {
            "count_found": count_found,
            "countReject": countReject,
            "count_1photon": count_1photon,
            "count_2photon": count_2photon,
            "count_morethan2": count_morethan2,
            "countPerfect": countPerfect,
            "count_withDiagonal": count_withDiagonal,
            "list_countij": list_countij,
            "list_ADU_sum": list_ADU_sum,
        }

        KT = self.KernelTypes(self, outputDict_initialised,
                              diagnostics=diagnostics, image_matrix=image_matrix_replace)

        print("-" * 30)
        print(f"Investigating clusters of the form {kernel_type}")

        if kernel_type == "single_pixel":
            # pRemoved denotes points that have been checked are removed
            outputDict, imMat_pRemoved = KT.single_pixel()
        elif kernel_type == "double_pixel":
            outputDict, imMat_pRemoved = KT.double_pixel()
        elif kernel_type == "triple_pixel":
            outputDict, imMat_pRemoved = KT.triple_pixel()
        elif kernel_type == "quadruple_pixel":
            outputDict, imMat_pRemoved = KT.quadruple_pixel()
        elif kernel_type == "quintuple_pixel":
            outputDict, imMat_pRemoved = KT.quintuple_pixel()
        elif kernel_type == "tp_line":
            outputDict, imMat_pRemoved = KT.tp_line()
        else:
            raise ValueError(f"Kernel type {kernel_type} not recognised")

        if report:
            print(f"Number of found elements: {outputDict['count_found']}")
            print(f"Number of found elements rejected: {outputDict['countReject']}")
            print(f"Number of 1 photon elements: {outputDict['count_1photon']}")
            print(f"Number of 2 photon elements: {outputDict['count_2photon']}")
            print(f"Number of elements with more than 2 photons: {outputDict["count_morethan2"]}")
            print(f"Number of perfect matches: {outputDict['countPerfect']}")
            print(f"Number of elements with diagonal: {outputDict['count_withDiagonal']}")
            print("-" * 30)

        return outputDict, imMat_pRemoved

    def display_kernel_type(self, kernel_type, diagnostics=False,
                            report=False,bins=300,image_matrix_replace=None):

        outputDict, imMat_pRemoved = self.check_kernel_type(kernel_type,
                                            diagnostics=diagnostics, report=report,
                                            image_matrix_replace=image_matrix_replace)
        list_ADU_sum = outputDict["list_ADU_sum"]

        def plotADU_hist():
            hist_values, bin_edges = np.histogram(list_ADU_sum, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.plot(bin_centers, hist_values)
            plt.yscale("log")
            plt.xlabel("ADU")
            plt.ylabel("Number of elements")
            plt.title(f"Histogram of ADU values for {kernel_type} kernel")
            plt.show()

        plotADU_hist()

    def kernelType_reducedMat(self,folder_filepath,filename,matrix_to_test=None):
        """
        Use all kernel types and reduce the matrix each time by feeding imCopy into the next kernel
        search. The function will save each step for investigation and comparison to ensure that the
        function is running as planned. The aim will be that the final matrix is only operated on as
        the reduced matrix with just images.
        :return:
        """

        kernel_dictionaries = kernelDict()
        output_dictionaries = {}

        if matrix_to_test is None:
            imMat_pRemoved = self.imMat.copy()
        else:
            imMat_pRemoved = matrix_to_test.copy()

        np.save(f"{folder_filepath}/{filename}_raw.npy", imMat_pRemoved)

        for i,kernels in enumerate(kernel_dictionaries.keys()):
            outputDict, imMat_pRemoved = self.check_kernel_type(kernels, diagnostics=False,report=True,image_matrix_replace=imMat_pRemoved)

            # Storing the individual output dictionary
            output_dictionaries[kernels] = outputDict

            # Save the intermediate image
            np.save(f"{folder_filepath}/{filename}_image{i}.npy", imMat_pRemoved)

        # Save the final Image and a text file with initialised parameters:
        np.save(f"{folder_filepath}/{filename}_final.npy",imMat_pRemoved)

        def log_params():
            txt_filepath = f"{folder_filepath}/{filename}_notes.txt"
            atf = Append_to_file(txt_filepath)
            app = atf.append
            app("-" * 30)
            app("Photon Countining Initialised Parameters:")
            app(f"index of interest = {self.index_of_interest}")
            app(f"no_p_adu_thr = {self.no_p_adu_thr}")
            app(f"sp_adu_thr = {self.sp_adu_thr}")
            app(f"dp_adu_thr = {self.dp_adu_thr}")
            app(f"howManySigma = {self.howManySigma}")

            app(f"Elements were removed with kernels in the following order:")
            app(f"{kernel_dictionaries.keys()}")

            for kernel in kernel_dictionaries.keys():
                app("-" * 30)
                app(f"{kernel}:")
                app("-" * 30)
                app(f"Number of found elements: {outputDict['count_found']}")
                app(f"Number of found elements rejected: {outputDict['countReject']}")
                app(f"Number of 1 photon elements: {outputDict['count_1photon']}")
                app(f"Number of 2 photon elements: {outputDict['count_2photon']}")
                app(f"Number of elements with more than 2 photons: {outputDict["count_morethan2"]}")
                app(f"Number of perfect matches: {outputDict['countPerfect']}")
                app(f"Number of elements with diagonal: {outputDict['count_withDiagonal']}")

        log_params()

        return output_dictionaries


    class KernelTypes:
        def __init__(self, parentClass, initialisedOutputDict,
                     diagnostics=False, image_matrix=None):
            """
            :param parentClass:
            :param initialisedOutputDict:
            :param diagnostics:
            :param image_matrix: If we want to impose an image matrix over this we can do so here
            """

            # Reinitialise relevant varriables from Photon Counting
            self.parent = parentClass

            if image_matrix is not None:
                self.imMat = image_matrix
            else:
                self.imMat = self.parent.imMat
            self.no_p_adu_thr = self.parent.no_p_adu_thr
            self.sp_adu_thr = self.parent.sp_adu_thr
            self.dp_adu_thr = self.parent.dp_adu_thr

            self.rowNum, self.colNum = self.imMat.shape
            # Create binary matrix as we are only searching for shape initially
            self.image_binary = np.where(self.imMat > 0, 1, 0)
            self.outputMat = np.zeros(self.imMat.shape)

            self.count_found = initialisedOutputDict["count_found"]
            self.countReject = initialisedOutputDict["countReject"]
            self.count_1photon = initialisedOutputDict["count_1photon"]
            self.count_2photon = initialisedOutputDict["count_2photon"]
            self.count_morethan2 = initialisedOutputDict["count_morethan2"]
            self.countPerfect = initialisedOutputDict["countPerfect"]
            self.count_withDiagonal = initialisedOutputDict["count_withDiagonal"]
            self.list_countij = initialisedOutputDict["list_countij"]
            self.list_ADU_sum = initialisedOutputDict["list_ADU_sum"]

            self.diagnostics = diagnostics

        def single_pixel(self):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            kdict = kernelDict()["single_pixel"]
            kernels = kdict["kernels"]
            masks = kdict["masks"]

            for kernel, mask in zip(kernels, masks):
                k_rows, k_cols = kernel.shape

                # Convolve the image
                for i in range(self.rowNum - k_rows + 1):
                    for j in range(self.colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                        if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                            continue

                        self.count_found += 1

                        image_copy[i+1,j+1] = 0

                        singlePixelVal = self.imMat[i + 1, j + 1]

                        self.list_ADU_sum.append(singlePixelVal)

                        if singlePixelVal < self.no_p_adu_thr:
                            self.countReject += 1
                            continue

                        # Perfect Match
                        if np.array_equal(convolvedArea, kernel):
                            self.countPerfect += 1
                        # Match with diagonal
                        elif check_diagonals(convolvedArea, kernel, mask):
                            self.count_withDiagonal += 1


                        if self.no_p_adu_thr < singlePixelVal <= self.sp_adu_thr:
                            self.list_countij.append([1, i + 1, j + 1])
                            self.count_1photon += 1
                        elif self.sp_adu_thr < singlePixelVal <= self.dp_adu_thr:
                            self.list_countij.append([2, i + 1, j + 1])
                            self.count_2photon += 1
                        else:
                            self.list_countij.append([3, i + 1, j + 1])
                            self.count_morethan2 += 1

                            if self.diagnostics:
                                print(f"Found a 3 photon hit at i={i + 1},j={j + 1}")

            output_dict = {
                "count_found": self.count_found,
                "countReject": self.countReject,
                "count_1photon": self.count_1photon,
                "count_2photon": self.count_2photon,
                "count_morethan2": self.count_morethan2,
                "countPerfect": self.countPerfect,
                "count_withDiagonal": self.count_withDiagonal,
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy

        def double_pixel(self):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            kdict = kernelDict()["double_pixel"]
            kernels = kdict["kernels"]
            masks = kdict["masks"]

            for kernel, mask in zip(kernels, masks):
                k_rows, k_cols = kernel.shape

                # Convolve the image
                for i in range(self.rowNum - k_rows + 1):
                    for j in range(self.colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                        if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                            continue

                        self.count_found += 1


                        if k_rows == 3:
                            # Horizontal case
                            # Value on the left
                            AVal = self.imMat[i + 1, j + 1]
                            # Value on the Right
                            BVal = self.imMat[i + 1, j + 2]

                            Aindexi,Aindexj = (i + 1,j + 1)
                            Bindexi,Bindexj = (i + 1,j + 2)

                        elif k_rows == 4:
                            # Vertical Case
                            AVal = self.imMat[i + 1, j + 1]
                            BVal = self.imMat[i + 2, j + 1]

                            Aindexi,Aindexj = (i + 1,j + 1)
                            Bindexi,Bindexj = (i + 2,j + 1)
                        else:
                            print("The kernel Matrix did not have 3 or 4 rows")
                            print("k_rows = ", k_rows, " k_cols = ", k_cols)
                            print("k_rows type", type(k_rows))
                            print(kernel)
                            raise ValueError


                        image_copy[Aindexi, Aindexj] = 0
                        image_copy[Bindexi, Bindexj] = 0
                        totVal = AVal + BVal

                        self.list_ADU_sum.append(totVal)

                        if totVal <= self.no_p_adu_thr:
                            self.countReject += 1
                            continue

                        # Perfect Match
                        if np.array_equal(convolvedArea, kernel):
                            self.countPerfect += 1
                        # Match with diagonal
                        elif check_diagonals(convolvedArea, kernel, mask):
                            self.count_withDiagonal += 1

                        elif totVal < self.sp_adu_thr:
                            self.count_1photon += 1
                            if AVal > BVal:
                                self.list_countij.append([1, Aindexi, Aindexj])
                            elif BVal > AVal:
                                self.list_countij.append([1, Bindexi, Bindexj])
                        elif totVal < self.dp_adu_thr:
                            self.count_2photon += 1
                            if AVal > BVal:
                                self.list_countij.append([2, Aindexi, Aindexj])
                            elif BVal > AVal:
                                self.list_countij.append([2, Bindexi, Bindexj])
                        elif totVal > self.dp_adu_thr:
                            self.count_morethan2 += 1
                            if self.diagnostics:
                                print(f"Found a 3 photon hit near i={i},j={j}:")
                                print(totVal)
                                print(self.imMat[i:i + k_rows, j:j + k_cols])

            output_dict = {
                "count_found": self.count_found,
                "countReject": self.countReject,
                "count_1photon": self.count_1photon,
                "count_2photon": self.count_2photon,
                "count_morethan2": self.count_morethan2,
                "countPerfect": self.countPerfect,
                "count_withDiagonal": self.count_withDiagonal,
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy

        def triple_pixel(self):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            kdict = kernelDict()["triple_pixel"]
            kernels = kdict["kernels"]
            masks = kdict["masks"]

            for kernel, mask in zip(kernels, masks):
                k_rows, k_cols = kernel.shape
                # Convolve the image
                for i in range(self.rowNum - k_rows + 1):
                    for j in range(self.colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                        if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                            continue

                        self.count_found += 1

                        nonZeroIndices = np.argwhere(kernel != 0)

                        dict_idx = {
                            "a": nonZeroIndices[0],
                            "b": nonZeroIndices[1],
                            "c": nonZeroIndices[2],
                        }

                        dict_vals = {}
                        totVal = 0
                        for key in dict_idx.keys():
                            dict_vals[key] = self.imMat[i+dict_idx[key][0], j+dict_idx[key][1]]
                            totVal += dict_vals[key]

                            # Removing captured points from the copy
                            image_copy[i + dict_idx[key][0], j + dict_idx[key][1]] = 0

                        self.list_ADU_sum.append(totVal)

                        if totVal < self.no_p_adu_thr:
                            if self.diagnostics:
                                print(totVal)
                            self.countReject += 1
                            continue

                        # Perfect Match
                        if np.array_equal(convolvedArea, kernel):
                            self.countPerfect += 1
                        # Match with diagonal
                        elif check_diagonals(convolvedArea, kernel, mask):
                            self.count_withDiagonal += 1


                        if totVal < self.sp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            max_key = keyOrderedList[0]

                            self.list_countij.append([1, i + dict_idx[max_key][0], j + + dict_idx[max_key][0]])
                            self.count_1photon += 1

                        elif totVal < self.dp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            key_1 = keyOrderedList[0]
                            key_2 = keyOrderedList[1]

                            self.list_countij.append([1, i + dict_idx[key_1][0], j + dict_idx[key_1][1]])
                            self.list_countij.append([1, i + dict_idx[key_2][0], j + dict_idx[key_2][1]])
                            self.count_2photon += 2

                            if self.diagnostics:
                                print("dp_adu_thr")
                                print(self.imMat[i:i + k_rows, j:j + k_cols])
                        else:
                            self.count_morethan2 += 1
                            if self.diagnostics:
                                print(f"Found a 3 photon hit near i={i},j={j}:")
                                print(totVal)
                                print(self.imMat[i:i + k_rows, j:j + k_cols])

            output_dict = {
                "count_found": self.count_found,
                "countReject": self.countReject,
                "count_1photon": self.count_1photon,
                "count_2photon": self.count_2photon,
                "count_morethan2": self.count_morethan2,
                "countPerfect": self.countPerfect,
                "count_withDiagonal": self.count_withDiagonal,
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy

        def quadruple_pixel(self):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            kdict = kernelDict()["quadruple_pixel"]

            kernels = kdict["kernels"]
            masks = kdict["masks"]

            for kernel, mask in zip(kernels, masks):
                k_rows, k_cols = kernel.shape

                # Convolve the image
                for i in range(self.rowNum - k_rows + 1):
                    for j in range(self.colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                        if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                            continue

                        self.count_found += 1

                        # Create dictionary with keys tl = top left, top right, bottom left etc.
                        dict_idx = {
                            "tl": [i + 1, j + 1],
                            "tr": [i + 1, j + 2],
                            "bl": [i + 2, j + 1],
                            "br": [i + 2, j + 2],
                        }

                        # initialise a dictionary of values with the same key
                        dict_vals = {}
                        totVal = 0
                        # Also initialise totVal to find the total of all 4 spots
                        for key in dict_idx.keys():
                            valOfKey = self.imMat[dict_idx[key][0], dict_idx[key][1]]
                            dict_vals[key] = valOfKey
                            totVal += valOfKey

                            # Removing captured points from the copy
                            image_copy[dict_idx[key][0], dict_idx[key][1]] = 0

                        if totVal < self.no_p_adu_thr:
                            self.countReject += 1
                            continue

                        # Perfect Match
                        if np.array_equal(convolvedArea, kernel):
                            self.countPerfect += 1
                        # Match with diagonal
                        elif check_diagonals(convolvedArea, kernel, mask):
                            self.count_withDiagonal += 1

                        self.list_ADU_sum.append(totVal)

                        if totVal < self.sp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            max_key = keyOrderedList[0]

                            self.list_countij.append([1, dict_idx[max_key][0], dict_idx[max_key][1]])
                            self.count_1photon += 1
                        elif totVal < self.dp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)

                            key_1 = keyOrderedList[0]
                            key_2 = keyOrderedList[1]

                            self.list_countij.append([1, dict_idx[key_1][0], dict_idx[key_1][1]])
                            self.list_countij.append([1, dict_idx[key_2][0], dict_idx[key_2][1]])

                            if self.diagnostics:
                                print("dp_adu_thr")
                                print(self.imMat[i:i + k_rows, j:j + k_cols])

                            self.count_2photon += 1
                        else:
                            if self.diagnostics:
                                print("more than dp_adu_thr")
                                print(self.imMat[i:i + k_rows, j:j + k_cols])
                            self.count_morethan2 += 1

            output_dict = {
                "count_found": self.count_found,
                "countReject": self.countReject,
                "count_1photon": self.count_1photon,
                "count_2photon": self.count_2photon,
                "count_morethan2": self.count_morethan2,
                "countPerfect": self.countPerfect,
                "count_withDiagonal": self.count_withDiagonal,
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy

        def quintuple_pixel(self):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            kdict = kernelDict()["quintuple_pixel"]
            kernels = kdict["kernels"]
            masks = kdict["masks"]

            for kernel, mask in zip(kernels, masks):
                k_rows, k_cols = kernel.shape
                # Convolve the image
                for i in range(self.rowNum - k_rows + 1):
                    for j in range(self.colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                        if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                            continue

                        self.count_found += 1

                        nonZeroIndices = np.argwhere(kernel != 0)

                        dict_idx = {
                            "a": nonZeroIndices[0],
                            "b": nonZeroIndices[1],
                            "c": nonZeroIndices[2],
                            "d": nonZeroIndices[3],
                            "e": nonZeroIndices[4],
                        }

                        dict_vals = {}
                        totVal = 0
                        for key in dict_idx.keys():
                            dict_vals[key] = self.imMat[i+dict_idx[key][0], j+dict_idx[key][1]]
                            totVal += dict_vals[key]

                            # Removing captured points from the copy
                            image_copy[i + dict_idx[key][0], j + dict_idx[key][1]] = 0

                        self.list_ADU_sum.append(totVal)

                        if totVal < self.no_p_adu_thr:
                            if self.diagnostics:
                                print(totVal)
                            self.countReject += 1
                            continue

                        # Perfect Match
                        if np.array_equal(convolvedArea, kernel):
                            self.countPerfect += 1
                        # Match with diagonal
                        elif check_diagonals(convolvedArea, kernel, mask):
                            self.count_withDiagonal += 1


                        if totVal < self.sp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            max_key = keyOrderedList[0]

                            self.list_countij.append([1, i + dict_idx[max_key][0], j + + dict_idx[max_key][0]])
                            self.count_1photon += 1

                        elif totVal < self.dp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            key_1 = keyOrderedList[0]
                            key_2 = keyOrderedList[1]

                            self.list_countij.append([1, i + dict_idx[key_1][0], j + dict_idx[key_1][1]])
                            self.list_countij.append([1, i + dict_idx[key_2][0], j + dict_idx[key_2][1]])
                            self.count_2photon += 2

                            if self.diagnostics:
                                print("dp_adu_thr")
                                print(self.imMat[i:i + k_rows, j:j + k_cols])
                        else:
                            self.count_morethan2 += 1
                            if self.diagnostics:
                                print(f"Found a 3 photon hit near i={i},j={j}:")
                                print(totVal)
                                print(self.imMat[i:i + k_rows, j:j + k_cols])

            output_dict = {
                "count_found": self.count_found,
                "countReject": self.countReject,
                "count_1photon": self.count_1photon,
                "count_2photon": self.count_2photon,
                "count_morethan2": self.count_morethan2,
                "countPerfect": self.countPerfect,
                "count_withDiagonal": self.count_withDiagonal,
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy

        def tp_line(self):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            kdict = kernelDict()["tp_line"]
            kernels = kdict["kernels"]
            masks = kdict["masks"]

            for kernel, mask in zip(kernels, masks):
                k_rows, k_cols = kernel.shape
                # Convolve the image
                for i in range(self.rowNum - k_rows + 1):
                    for j in range(self.colNum - k_cols + 1):
                        # Consider areas of the same size as the kernel:
                        convolvedArea = self.image_binary[i:i + k_rows, j:j + k_cols]

                        if not (np.array_equal(convolvedArea, kernel) or check_diagonals(convolvedArea, kernel, mask)):
                            continue

                        self.count_found += 1

                        nonZeroIndices = np.argwhere(kernel != 0)

                        dict_idx = {
                            "a": nonZeroIndices[0],
                            "b": nonZeroIndices[1],
                            "c": nonZeroIndices[2],
                        }

                        dict_vals = {}
                        totVal = 0
                        for key in dict_idx.keys():
                            dict_vals[key] = self.imMat[i+dict_idx[key][0], j+dict_idx[key][1]]
                            totVal += dict_vals[key]

                            # Removing captured points from the copy
                            image_copy[i + dict_idx[key][0], j + dict_idx[key][1]] = 0

                        self.list_ADU_sum.append(totVal)

                        if totVal < self.no_p_adu_thr:
                            if self.diagnostics:
                                print(totVal)
                            self.countReject += 1
                            continue

                        # Perfect Match
                        if np.array_equal(convolvedArea, kernel):
                            self.countPerfect += 1
                        # Match with diagonal
                        elif check_diagonals(convolvedArea, kernel, mask):
                            self.count_withDiagonal += 1


                        if totVal < self.sp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            max_key = keyOrderedList[0]

                            self.list_countij.append([1, i + dict_idx[max_key][0], j + + dict_idx[max_key][0]])
                            self.count_1photon += 1

                        elif totVal < self.dp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            key_1 = keyOrderedList[0]
                            key_2 = keyOrderedList[1]

                            self.list_countij.append([1, i + dict_idx[key_1][0], j + dict_idx[key_1][1]])
                            self.list_countij.append([1, i + dict_idx[key_2][0], j + dict_idx[key_2][1]])
                            self.count_2photon += 2

                            if self.diagnostics:
                                print("dp_adu_thr")
                                print(self.imMat[i:i + k_rows, j:j + k_cols])
                        else:
                            self.count_morethan2 += 1
                            if self.diagnostics:
                                print(f"Found a 3 photon hit near i={i},j={j}:")
                                print(totVal)
                                print(self.imMat[i:i + k_rows, j:j + k_cols])

            output_dict = {
                "count_found": self.count_found,
                "countReject": self.countReject,
                "count_1photon": self.count_1photon,
                "count_2photon": self.count_2photon,
                "count_morethan2": self.count_morethan2,
                "countPerfect": self.countPerfect,
                "count_withDiagonal": self.count_withDiagonal,
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy


if __name__ == "__main__":
    def test_TypePhoton():

        hms_thr = 1.5
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=hms_thr)
        reducedMat = TestImages().image_8_emission_lines(hms_thr*pc.sigmaPedestal)

        pc.display_kernel_type("single_pixel",report=True,bins=300,image_matrix_replace=reducedMat)
        pc.display_kernel_type("double_pixel", report=True,bins=300,image_matrix_replace=reducedMat)
        pc.display_kernel_type("triple_pixel", report=True,bins=300,image_matrix_replace=reducedMat)
        pc.display_kernel_type("quadruple_pixel", report=True,bins=300,image_matrix_replace=reducedMat)

    # test_TypePhoton()

    def unitTest(printMatBetween=True,matToTest=None):
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=2)

        if matToTest is None:
            ut_mat = TestImages().diagonals()
        else:
            ut_mat = matToTest

        # renaming this so that we start using the repeated matrix
        imMat_pRemoved = ut_mat.copy()

        if printMatBetween:
            plt.imshow(ut_mat)
            plt.show()

        for ktype in ["single_pixel","double_pixel","triple_pixel","quadruple_pixel"]:
            outputDict, imMat_pRemoved = pc.check_kernel_type(ktype, report=True, image_matrix_replace=imMat_pRemoved)

            if printMatBetween:
                plt.imshow(imMat_pRemoved)
                plt.show()

        if not printMatBetween:
            plt.imshow(ut_mat)
            plt.show()

    # unitTest(True)

    def compareFaveLittleSpot():
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=2)
        sigma = pc.sigmaPedestal
        mean = pc.meanPedestal

        thr_sigma_1 = 2
        thr_sigma_2 = 3

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(TestImages().image8_cluster(thr_after_mean_removed=thr_sigma_1 * sigma, mean=mean), cmap='hot'), plt.title(
            f"Image Thresholded above {thr_sigma_1} sigma")
        plt.subplot(1, 2, 2), plt.imshow(TestImages().image8_cluster(thr_after_mean_removed=thr_sigma_2 * sigma, mean=mean), cmap='hot'), plt.title(
            f"Image 8 Thresholded above {thr_sigma_2} sigma")
        plt.show()

    # compareFaveLittleSpot()

    def unitTest_imageOutputs(openIms=False):
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=2,)
        ut_filepath = r"/data_logs/image_matrices/unit_test"

        testMat = TestImages().diagonals()
        pc.kernelType_reducedMat(folder_filepath=ut_filepath,filename="unit_test_1",matrix_to_test=testMat)

        if openIms:
            for file_name in os.listdir(ut_filepath):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(ut_filepath, file_name)
                    data = np.load(file_path)
                    plt.imshow(data)
                    plt.title(file_name)
                    plt.show()


    unitTest_imageOutputs(True)

    pass
