import os
from tools import *
from testImages import *
import time
from datetime import datetime

im8_filepath = r"old_logs_and_stored_variables/v1/data_logs\image_matrices\image_8"




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

        quint_kernel_5 = np.array([[0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 0],
                                   [0, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0]])
        quint_kernel_6 = np.rot90(quint_kernel_5)
        quint_kernel_7 = np.rot90(quint_kernel_6)
        quint_kernel_8 = np.rot90(quint_kernel_7)

        quint_mask_1 = np.array([[0, 1, 1, 1, 0],
                                 [1, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1],
                                 [0, 0, 1, 1, 0]])
        quint_mask_2 = np.rot90(quint_mask_1)
        quint_mask_3 = np.rot90(quint_mask_2)
        quint_mask_4 = np.rot90(quint_mask_3)

        quint_mask_5 = np.array([[0, 0, 1, 1, 0],
                                 [0, 1, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [0, 1, 1, 1, 0]])
        quint_mask_6 = np.rot90(quint_mask_5)
        quint_mask_7 = np.rot90(quint_mask_6)
        quint_mask_8 = np.rot90(quint_mask_7)

        return {
            "kernels": [quint_kernel_1, quint_kernel_2, quint_kernel_3, quint_kernel_4, quint_kernel_5, quint_kernel_6,
                        quint_kernel_7, quint_kernel_8],
            "masks": [quint_mask_1, quint_mask_2, quint_mask_3, quint_mask_4, quint_mask_5, quint_mask_6, quint_mask_7,
                      quint_mask_8],
        }

    def tp_line():
        tp_line_kernel_1 = np.array([[0, 0, 0, 0, 0],
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

    ll_kernel_1 = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]])
    ll_mask_1 = np.array([[0, 1, 1, 1, 0],
                          [1, 0, 0, 0, 1],
                          [0, 1, 1, 0, 1],
                          [0, 0, 0, 1, 0]])

    t_kernel_1 = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0]])

    t_mask_1 = np.array([[0, 1, 1, 1, 0],
                         [1, 0, 0, 0, 1],
                         [0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 0]])

    zigzag_kernel_1 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0]])
    zigzag_mask_1 = np.array([[0, 0, 1, 1, 0],
                              [0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0],
                              [0, 1, 1, 0, 0]])

    def kernel_flip_rotate(kernel_1, mask_1):
        kernel_2 = np.rot90(kernel_1)
        kernel_3 = np.rot90(kernel_2)
        kernel_4 = np.rot90(kernel_3)

        kernel_5 = np.flipud(kernel_1)
        kernel_6 = np.rot90(kernel_5)
        kernel_7 = np.rot90(kernel_6)
        kernel_8 = np.rot90(kernel_7)

        mask_2 = np.rot90(mask_1)
        mask_3 = np.rot90(mask_2)
        mask_4 = np.rot90(mask_3)

        mask_5 = np.flipud(mask_1)
        mask_6 = np.rot90(mask_5)
        mask_7 = np.rot90(mask_6)
        mask_8 = np.rot90(mask_7)

        return {
            "kernels": [kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7,
                        kernel_8],
            "masks": [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8],
        }

    return {
        "single_pixel": single_pixel(),
        "double_pixel": double_pixel(),
        "triple_pixel": triple_pixel(),
        "quadruple_pixel": quadruple_pixel(),
        # "quintuple_pixel": quintuple_pixel(),
        # "tp_line": tp_line(),
        # "long_L": kernel_flip_rotate(ll_kernel_1, ll_mask_1),
        # "t_shape": kernel_flip_rotate(t_kernel_1, t_mask_1),
        # "zigzag": kernel_flip_rotate(zigzag_kernel_1, zigzag_mask_1),
    }


def shapesDict():
    kernel_dict = kernelDict()
    dictionary_shapes = {}

    for key in kernel_dict.keys():
        dict_of_key = kernel_dict[key]
        kernelList = dict_of_key["kernels"]
        maskList = dict_of_key["masks"]

        for kernel, mask in zip(kernelList, maskList):

            shape_kernel = kernel.shape

            if shape_kernel not in dictionary_shapes:
                dictionary_shapes[shape_kernel] = {}  # Initialize an empty dictionary

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
    def __init__(self, indexOfInterest, no_photon_adu_thr=80, sp_adu_thr=180, dp_adu_thr=240,
                 tp_adu_thr=400, quad_p_adu_thr=550, quint_p_adu_thr=700,
                 removeRows0To_=0, howManySigma_thr=2, ):

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
        self.tp_adu_thr = tp_adu_thr
        self.quad_p_adu_thr = quad_p_adu_thr
        self.quin_p_adu_thr = quint_p_adu_thr

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

    def check_kernel_type(self, kernel_type, diagnostics=False,
                          report=False, image_matrix_replace=None):
        # Initialise Counts
        count_found = 0
        countReject = 0
        count_1photon = 0
        count_2photon = 0
        count_morethan2 = 0

        list_countij = []
        list_ADU_sum = []

        outputDict_initialised = {
            "count_found": count_found,
            "countReject": countReject,
            "count_1photon": count_1photon,
            "count_2photon": count_2photon,
            "count_morethan2": count_morethan2,
            "list_countij": list_countij,
            "list_ADU_sum": list_ADU_sum,
        }

        KT = self.KernelTypes(self, outputDict_initialised,
                              diagnostics=diagnostics, image_matrix=image_matrix_replace)

        print("-" * 30)
        print(f"Investigating clusters of the form {kernel_type}")

        if kernel_type in kernelDict():
            # pRemoved denotes points that have been checked are removed
            outputDict, imMat_pRemoved = KT.generalised_fct(kernel_type)
        else:
            raise ValueError(f"Kernel type {kernel_type} not recognised")

        if report:
            print(f"Number of found elements: {outputDict['count_found']}")
            print(f"Number of found elements rejected: {outputDict['countReject']}")
            print(f"Number of 1 photon elements: {outputDict['count_1photon']}")
            print(f"Number of 2 photon elements: {outputDict['count_2photon']}")
            print(f"Number of elements with more than 2 photons: {outputDict["count_morethan2"]}")
            print("-" * 30)

        return outputDict, imMat_pRemoved

    def display_kernel_type(self, kernel_type, diagnostics=False,
                            report=False, bins=300, image_matrix_replace=None):

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

    def kernelType_reducedMat(self, folder_filepath, filename, matrix_to_test=None, diagnostics=False, ):
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

        for i, kernels in enumerate(kernel_dictionaries.keys()):
            outputDict, imMat_pRemoved = self.check_kernel_type(kernels, diagnostics=diagnostics, report=True,
                                                                image_matrix_replace=imMat_pRemoved)

            # Storing the individual output dictionary
            output_dictionaries[kernels] = outputDict

            # Save the intermediate image
            np.save(f"{folder_filepath}/{filename}_image{i}.npy", imMat_pRemoved)

        # Save the final Image and a text file with initialised parameters:
        np.save(f"{folder_filepath}/{filename}_final.npy", imMat_pRemoved)

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

        log_params()

        return output_dictionaries

    def efficient_imageScan(self, dictionary_shapes=None, diagnostics=False,
                            report=False, image_matrix_replace=None, intermediary_matrices_arg_dict=None):
        """

        :param dictionary_shapes:
        :param diagnostics:
        :param report:
        :param image_matrix_replace:
        :param intermediary_matrices_arg_dict: Dictionary with "folderpath" and "filename"
        :return:
        """

        startTime = time.time()

        if dictionary_shapes is None:
            dictionary_shapes = shapesDict()

        # Dictionary of shapes has a series of keys: shape --> kernel_type --> "kernels","masks"
        # print(dictionary_shapes)

        # Extract unique keys from all dictionaries
        unique_keys = kernelDict().keys()
        # Initilaise output dict for each label
        output_dictionaries = {}
        for key in unique_keys:
            # Initialise Counts
            count_found = 0
            countReject = 0
            count_1photon = 0
            count_2photon = 0
            count_morethan2 = 0

            list_countij = []
            list_ADU_sum = []

            output_dictionaries[key] = {
                "count_found": count_found,
                "countReject": countReject,
                "count_1photon": count_1photon,
                "count_2photon": count_2photon,
                "count_morethan2": count_morethan2,
                "list_countij": list_countij,
                "list_ADU_sum": list_ADU_sum,
            }

        if image_matrix_replace is None:
            imMat_pRemoved = self.imMat.copy()
        else:
            imMat_pRemoved = image_matrix_replace

        if intermediary_matrices_arg_dict is not None:
            folderpath = intermediary_matrices_arg_dict["folderpath"]
            filename = intermediary_matrices_arg_dict["filename"]

            np.save(f"{folderpath}/{filename}_raw.npy", imMat_pRemoved)

        rowNum, colNum = imMat_pRemoved.shape
        im_binary = np.where(self.imMat > 0, 1, 0)

        # Create a matrix that stores checked points
        checked_mat = np.zeros((rowNum, colNum))

        for number, shape_tuple in enumerate(dictionary_shapes.keys()):

            print(f"Scanning with {shape_tuple}")

            dict_kernel_type = dictionary_shapes[shape_tuple]

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

                            # Marking Checked Squares

                            outputDict_kt["count_found"] += 1
                            numPoints_kernel = len(idxKernel)
                            dict_idx = {}
                            for point_number in range(numPoints_kernel):
                                dict_idx[point_number] = idxKernel[point_number]

                            for idx in idxKernel:
                                checked_mat[i + idx[0], j + idx[1]] = 1
                            for idx in idxMask:
                                checked_mat[i + idx[0], j + idx[1]] = 1

                            dict_vals = {}
                            totVal = 0

                            for key in dict_idx.keys():
                                dict_vals[key] = self.imMat[i + dict_idx[key][0], j + dict_idx[key][1]]
                                totVal += dict_vals[key]

                                # Removing captured points from the copy
                                imMat_pRemoved[i + dict_idx[key][0], j + dict_idx[key][1]] = 0

                            outputDict_kt["list_ADU_sum"].append(totVal)

                            if totVal < self.no_p_adu_thr:
                                outputDict_kt["countReject"] += 1
                                continue

                            if totVal < self.sp_adu_thr:
                                keyOrderedList = sorted_keys_by_value(dict_vals)
                                max_key = keyOrderedList[0]

                                outputDict_kt["list_countij"].append(
                                    [1, i + dict_idx[max_key][0], j + + dict_idx[max_key][0]])
                                outputDict_kt["count_1photon"] += 1
                            elif totVal < self.dp_adu_thr:
                                keyOrderedList = sorted_keys_by_value(dict_vals)

                                if numPoints_kernel == 1:
                                    outputDict_kt["list_countij"].append([2, i + dict_idx[0][0], j + dict_idx[0][1]])
                                else:
                                    key_1 = keyOrderedList[0]
                                    key_2 = keyOrderedList[1]

                                    outputDict_kt["list_countij"].append(
                                        [1, i + dict_idx[key_1][0], j + dict_idx[key_1][1]])
                                    outputDict_kt["list_countij"].append(
                                        [1, i + dict_idx[key_2][0], j + dict_idx[key_2][1]])
                                outputDict_kt["count_2photon"] += 2

                                if diagnostics:
                                    print("dp_adu_thr")
                                    print(self.imMat[i:i + k_rows, j:j + k_cols])
                            else:
                                outputDict_kt["count_morethan2"] += 1
                                if diagnostics:
                                    print(f"Found a 3 photon hit near i={i},j={j}:")
                                    print(totVal)
                                    print(self.imMat[i:i + k_rows, j:j + k_cols])

            if intermediary_matrices_arg_dict is not None:
                folderpath = intermediary_matrices_arg_dict["folderpath"]
                filename = intermediary_matrices_arg_dict["filename"]
                tuple_str = "_".join(str(x) for x in shape_tuple)
                np.save(f"{folderpath}/{filename}_{tuple_str}.npy", imMat_pRemoved)

        endtime = time.time()
        function_time = endtime - startTime
        minutes, seconds = divmod(function_time, 60)
        print(f"efficient_imagescan function runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

        if intermediary_matrices_arg_dict is not None:
            folderpath = intermediary_matrices_arg_dict["folderpath"]
            filename = intermediary_matrices_arg_dict["filename"]

            np.save(f"{folderpath}/{filename}_final.npy", imMat_pRemoved)

            np.save(f"{folderpath}/{filename}_checkedPoints.npy", checked_mat)

            def log_params():
                txt_filepath = f"{folderpath}/{filename}_notes.txt"
                atf = Append_to_file(txt_filepath)
                app = atf.append
                app("-" * 30)
                app(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                app(f"efficient_imagescan function runtime: {int(minutes)} minutes and {seconds:.2f} seconds")
                app("-" * 30)
                app("Photon Countining Initialised Parameters:")
                app(f"index of interest = {self.index_of_interest}")
                app(f"no_p_adu_thr = {self.no_p_adu_thr}")
                app(f"sp_adu_thr = {self.sp_adu_thr}")
                app(f"dp_adu_thr = {self.dp_adu_thr}")
                app(f"howManySigma = {self.howManySigma}")

                for kernel_type_ in unique_keys:
                    output_Dict = output_dictionaries[kernel_type_]
                    app("-" * 30)
                    app(f"{kernel_type_}:")
                    app("-" * 30)
                    app(f"Number of found elements: {output_Dict['count_found']}")
                    app(f"Number of found elements rejected: {output_Dict['countReject']}")
                    app(f"Number of 1 photon elements: {output_Dict['count_1photon']}")
                    app(f"Number of 2 photon elements: {output_Dict['count_2photon']}")
                    app(f"Number of elements with more than 2 photons: {output_Dict["count_morethan2"]}")

            log_params()

        if report:

            for kernel in unique_keys:
                outputDict = output_dictionaries[kernel]
                print("-" * 30)
                print(f"{kernel}:")
                print("-" * 30)
                print(f"Number of found elements: {outputDict['count_found']}")
                print(f"Number of found elements rejected: {outputDict['countReject']}")
                print(f"Number of 1 photon elements: {outputDict['count_1photon']}")
                print(f"Number of 2 photon elements: {outputDict['count_2photon']}")
                print(f"Number of elements with more than 2 photons: {outputDict["count_morethan2"]}")

        return output_dictionaries, imMat_pRemoved

    def operateOnIslands(self, image_matrix_replace=None, plot_checkedMat=False, diagnosticPrint=False):

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
                    elif totVal < self.sp_adu_thr:
                        numPhotons = 1
                    elif totVal < self.dp_adu_thr:
                        numPhotons = 2
                    elif totVal < self.tp_adu_thr:
                        numPhotons = 3
                    elif totVal < self.quad_p_adu_thr:
                        numPhotons = 4
                    elif totVal < self.quin_p_adu_thr:
                        numPhotons = 5
                    else:
                        print(f"There were more than 5 photons i~{i}, j~{j}, totVal = {totVal}")
                        results_dict["number_higher_than_capture"] += 1
                        continue

                    results_dict["number_of_photons"] += numPhotons

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
                                print(f"Photon at (i,j) = {ij_tuple}")

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
            self.list_countij = initialisedOutputDict["list_countij"]
            self.list_ADU_sum = initialisedOutputDict["list_ADU_sum"]

            self.diagnostics = diagnostics

        def generalised_fct(self, kernel_type):

            # Initialising a copy of the image matrix to have elements removed once counted
            image_copy = self.imMat.copy()

            if kernel_type not in kernelDict():
                raise ValueError(f"Kernel type {kernel_type} not recognised")

            kdict = kernelDict()[kernel_type]
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

                        howManyPoints = len(nonZeroIndices)

                        dict_idx = {}
                        for point_number in range(howManyPoints):
                            dict_idx[point_number] = nonZeroIndices[point_number]

                        if self.diagnostics:
                            print(dict_idx)

                        dict_vals = {}
                        totVal = 0
                        for key in dict_idx.keys():
                            dict_vals[key] = self.imMat[i + dict_idx[key][0], j + dict_idx[key][1]]
                            totVal += dict_vals[key]

                            # Removing captured points from the copy
                            image_copy[i + dict_idx[key][0], j + dict_idx[key][1]] = 0

                        self.list_ADU_sum.append(totVal)

                        if totVal < self.no_p_adu_thr:
                            if self.diagnostics:
                                print(totVal)
                            self.countReject += 1
                            continue

                        if totVal < self.sp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)
                            max_key = keyOrderedList[0]

                            self.list_countij.append([1, i + dict_idx[max_key][0], j + + dict_idx[max_key][0]])
                            self.count_1photon += 1

                        elif totVal < self.dp_adu_thr:
                            keyOrderedList = sorted_keys_by_value(dict_vals)

                            if howManyPoints == 1:
                                self.list_countij.append([2, i + dict_idx[0][0], j + dict_idx[0][1]])
                            else:
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
                "list_countij": self.list_countij,
                "list_ADU_sum": self.list_ADU_sum,
            }

            return output_dict, image_copy


class Unit_testing:

    @staticmethod
    def unitTest1(num_photons=1000, matrix_size=(100, 100), mean_adu=150, std_adu=10,
                  returnJustImage=False, seed=125):
        spcTrain = SPC_Train_images(2)

        unit_test_mat = spcTrain.createTestData(num_photons, matrix_size=matrix_size, mean_adu=mean_adu,
                                                std_adu=std_adu,
                                                returnJustImage=returnJustImage, seed=seed)

        adu_thr = [
            80,  # no_photon_adu_thr
            180,  # sp_adu_thr
            240,  # dp_adu_thr
            400,  # tp_adu_thr
            550,  # quad_p_adu_thr
            700,  # quint_p_adu_thr
        ]

        def loss_function(params):
            pc_eng = PhotonCounting(8, no_photon_adu_thr=params[0], sp_adu_thr=params[1], dp_adu_thr=params[2],
                                    tp_adu_thr=params[3], quad_p_adu_thr=params[4], quint_p_adu_thr=params[5],
                                    removeRows0To_=0, howManySigma_thr=2, )

            results_dict = pc_eng.operateOnIslands(image_matrix_replace=unit_test_mat, diagnosticPrint=True)

            numCaptured = results_dict["number_of_photons"]
            num_above_capture = results_dict["number_higher_than_capture"]

        loss_function(adu_thr)


if __name__ == "__main__":

    Unit_testing().unitTest1()


    def test_TypePhoton():

        hms_thr = 2
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=hms_thr)
        reducedMat = TestImages().image_8_emission_lines(hms_thr * pc.sigmaPedestal)

        # pc.display_kernel_type("single_pixel", report=True, bins=300, image_matrix_replace=reducedMat)
        # pc.display_kernel_type("double_pixel", report=True, bins=300, image_matrix_replace=reducedMat)
        # pc.display_kernel_type("triple_pixel", report=True, bins=300, image_matrix_replace=reducedMat)
        # pc.display_kernel_type("quadruple_pixel", report=True, bins=300, image_matrix_replace=reducedMat)

        # pc.display_kernel_type("quintuple_pixel", report=True, bins=300)
        # pc.display_kernel_type("tp_line", report=True, bins=300)
        pc.display_kernel_type("zigzag", report=True, bins=300)
        pc.display_kernel_type()


    # test_TypePhoton()

    def unitTest(printMatBetween=True, matToTest=None):

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

        for ktype in ["single_pixel", "double_pixel", "triple_pixel", "quadruple_pixel"]:
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
        plt.subplot(1, 2, 1), plt.imshow(
            TestImages().image8_cluster(thr_after_mean_removed=thr_sigma_1 * sigma, mean=mean), cmap='hot'), plt.title(
            f"Image Thresholded above {thr_sigma_1} sigma")
        plt.subplot(1, 2, 2), plt.imshow(
            TestImages().image8_cluster(thr_after_mean_removed=thr_sigma_2 * sigma, mean=mean), cmap='hot'), plt.title(
            f"Image 8 Thresholded above {thr_sigma_2} sigma")
        plt.show()


    # compareFaveLittleSpot()

    def unitTest_imageOutputs(openIms=False):

        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=80, howManySigma_thr=2, )
        ut_filepath = r"old_logs_and_stored_variables/v1/data_logs\image_matrices\unit_test"

        testMat = TestImages().diagonals()
        pc.kernelType_reducedMat(folder_filepath=ut_filepath, filename="unit_test_2", matrix_to_test=testMat,
                                 diagnostics=True)

        if openIms:
            for file_name in os.listdir(ut_filepath):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(ut_filepath, file_name)
                    data = np.load(file_path)
                    plt.imshow(data)
                    plt.title(file_name)
                    plt.show()


    # unitTest_imageOutputs(True)

    def test_efficient_scan():

        hms_thr = 2
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=30, howManySigma_thr=hms_thr)

        save_dict = {
            "folderpath": im8_filepath,
            "filename": "test3",
        }

        pc.efficient_imageScan(report=True, intermediary_matrices_arg_dict=None, diagnostics=False)


    # test_efficient_scan()

    def test_islands():
        pc = PhotonCounting(indexOfInterest=8, no_photon_adu_thr=30, howManySigma_thr=2)

        testMat = TestImages().image_8_post_kernels()
        testMat = pc.imMat
        testMat[:, 0:3] = 0
        # testMat[0:200, :] = 0

        rdict = pc.operateOnIslands(image_matrix_replace=testMat)

        # print(rdict)

        plt.imshow(testMat)
        plt.show()


    # test_islands()

    pass
