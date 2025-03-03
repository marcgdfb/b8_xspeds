from calibrateGeometry import *
from spc_engine_v3 import *
from collections import Counter
import time


# TODO: Make it clear in documentation that the mean is subtracted from the result. Do this in readme?

# TODO: Consider the probability given the gaussian of having

# TODO: think about what's going to be optimising :
# Initial image threshold, no photon some adu thr, 1 photon adu sum, 2 photon adu sum
# should the sum be different for each number

# TODO: consider giving i,j not to check and searching simultaneously for like sized shapes


class Spectrum:
    def __init__(self, indexOfInterest,
                 crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_camera_spherical, sp_adu_thr=180, dp_adu_thr=240, noPhoton_adu_thr=50,
                 removeTopRows=0,
                 how_many_sigma=0):

        def printVar():
            print("-" * 30)
            print("Spectrum Class Initialised with:")
            print("indexOfInterest = ", indexOfInterest)
            print("crystal_pitch = ", crystal_pitch)
            print("crystal_roll = ", crystal_roll)
            print("camera_pitch = ", camera_pitch)
            print("camera_roll = ", camera_roll)
            print("r_camera_spherical = ", r_camera_spherical)
            print("noPhoton_adu_thr = ", noPhoton_adu_thr)
            print("sp_adu_thr = ", sp_adu_thr)
            print("dp_adu_thr = ", dp_adu_thr)
            print("removeTopRows = ", removeTopRows)
            print("how_many_sigma = ", how_many_sigma)

        printVar()
        self.indexOfInterest = indexOfInterest
        self.noP_thresh = noPhoton_adu_thr
        self.sp_thresh = sp_adu_thr
        self.dp_thresh = dp_adu_thr

        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical

        self.removeTopRows = removeTopRows
        self.how_many_sigma = how_many_sigma

    def multiPixelSpectrum(self, band_width=2, methodList=None,
                           spectrumTitle=None,
                           plotSpectrum=False, logarithmic=False,
                           intensity_arb_unit=False, plotEachSubSpectrum=False,
                           intermediary_matrices_arg_dict=None
                           ):
        """
        :param band_width:
        :param methodList:
        :param spectrumTitle:
        :param plotSpectrum:
        :param logarithmic:
        :param intensity_arb_unit:
        :param plotEachSubSpectrum:
        :param intermediary_matrices_arg_dict: A dictionary with folder path, filename
        :return:
        """

        startTime = time.time()

        if spectrumTitle is None:
            howManySigmaTitle = f'\nConsidering values {self.how_many_sigma} above the mean'
            thresholds_title = f'\nNo Photons < {self.noP_thresh},SP < {self.sp_thresh},DP < {self.dp_thresh}'
            spectrumTitle = f"Photon Energy Spectrum with Multi Photon Hits Version 2" + howManySigmaTitle + thresholds_title

        if methodList is None:
            methodList = list(kdic.keys())

        spc_engine = PhotonCounting(indexOfInterest=self.indexOfInterest,
                                    no_photon_adu_thr=self.noP_thresh, sp_adu_thr=self.sp_thresh,
                                    dp_adu_thr=self.dp_thresh,
                                    removeRows0To_=self.removeTopRows,
                                    howManySigma_thr=self.how_many_sigma, )
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)
        energyDict = {}
        output_dictionaries = {}

        imMat_pRemoved = spc_engine.imMat.copy()

        if intermediary_matrices_arg_dict is not None:
            folderpath = intermediary_matrices_arg_dict["folderpath"]
            filename = intermediary_matrices_arg_dict["filename"]

            np.save(f"{folderpath}/{filename}_raw.npy", imMat_pRemoved)

        for number,method in enumerate(methodList):
            output_dictionary, imMat_pRemoved = spc_engine.check_kernel_type(kernel_type=method, report=True,image_matrix_replace=imMat_pRemoved)
            listCountij = output_dictionary["list_countij"]
            energyList = self.lists_energy(listCountij, braggEngine_init=bragg_engine)
            energyDict[method] = energyList

            output_dictionaries[method] = output_dictionary

            if intermediary_matrices_arg_dict is not None:
                folderpath = intermediary_matrices_arg_dict["folderpath"]
                filename = intermediary_matrices_arg_dict["filename"]
                np.save(f"{folderpath}/{filename}_{number}.npy", imMat_pRemoved)

        endtime = time.time()
        function_time = endtime - startTime
        minutes, seconds = divmod(function_time, 60)

        print(f"multiPixelSpectrum function runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

        if intermediary_matrices_arg_dict is not None:
            folderpath = intermediary_matrices_arg_dict["folderpath"]
            filename = intermediary_matrices_arg_dict["filename"]

            np.save(f"{folderpath}/{filename}_final.npy", imMat_pRemoved)

            def log_params():
                txt_filepath = f"{folderpath}/{filename}_notes.txt"
                atf = Append_to_file(txt_filepath)
                app = atf.append
                app("-" * 30)
                app(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                app(f"multiPixelSpectrum function runtime: {int(minutes)} minutes and {seconds:.2f} seconds")
                app("-" * 30)
                app("Photon Countining Initialised Parameters:")
                app(f"index of interest = {self.indexOfInterest}")
                app(f"no_p_adu_thr = {self.noP_thresh}")
                app(f"sp_adu_thr = {self.sp_thresh}")
                app(f"dp_adu_thr = {self.dp_thresh}")
                app(f"howManySigma = {self.how_many_sigma}")

                app(f"Elements were removed with kernels in the following order:")
                app(f"{methodList}")

                for kernel in methodList:
                    outputDict = output_dictionaries[kernel]

                    app("-" * 30)
                    app(f"{kernel}:")
                    app("-" * 30)
                    app(f"Number of found elements: {outputDict['count_found']}")
                    app(f"Number of found elements rejected: {outputDict['countReject']}")
                    app(f"Number of 1 photon elements: {outputDict['count_1photon']}")
                    app(f"Number of 2 photon elements: {outputDict['count_2photon']}")
                    app(f"Number of elements with more than 2 photons: {outputDict["count_morethan2"]}")

            log_params()



        if plotEachSubSpectrum:
            for method in methodList:
                energyList = energyDict[method]
                main_title = f'Photon Energy Spectrum with {method} Hits'
                howManySigmaTitle = f'Considering values {self.how_many_sigma} above the mean'
                thresholds_title = f'No Photons < {self.noP_thresh},SP < {self.sp_thresh},DP < {self.dp_thresh}'

                specTitleMethod = main_title + '\n' + howManySigmaTitle + '\n' + thresholds_title

                self.plotSpectrum(energyList, band_width,specTitleMethod,
                                  intensity_arb_unit, logarithmic)

        if plotSpectrum:
            energyList = []
            for method in methodList:
                energyList.extend(energyDict[method])
            self.plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit, logarithmic)

    def spectrumSpecificKernelType(self, band_width=2, method_string="single_pixel",
                                   spectrumTitle=None,
                                   plotSpectrum=False, logarithmic=False, intensity_arb_unit=False):
        if spectrumTitle is None:
            spectrumTitle = f"Photon Energy Spectrum with {method_string} method with vals above {self.how_many_sigma} sigma"

        spc_engine = PhotonCounting(indexOfInterest=self.indexOfInterest, sp_adu_thr=self.sp_thresh,
                                    dp_adu_thr=self.dp_thresh,
                                    no_photon_adu_thr=self.noP_thresh,
                                    removeRows0To_=self.removeTopRows,
                                    howManySigma_thr=self.how_many_sigma, )
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

        print("-" * 30)
        output_dictionary, imMat_pRemoved = spc_engine.check_kernel_type(method_string, report=True)
        listCountij = output_dictionary["list_countij"]
        count_occurrences = Counter(countij[0] for countij in listCountij)
        print(f"Count Occurences for {method_string} Hits")
        print(count_occurrences)

        energyList = self.lists_energy(listCountij, braggEngine_init=bragg_engine)

        if plotSpectrum:
            self.plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit, logarithmic)

    @staticmethod
    def plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit=False, logarithmic=False):

        energyBins = np.arange(min(energyList), max(energyList) + band_width, band_width)
        photonEnergies = np.array(energyList)
        count, bins_edges = np.histogram(photonEnergies, energyBins)

        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        plt.figure(figsize=(10, 6))
        if intensity_arb_unit:
            countIntensity = []
            for countE, center in zip(count, bin_centers):
                countIntensity.append(countE * center)
            countIntensity = np.array(countIntensity)
            plt.plot(bin_centers, countIntensity, linestyle='-', color='b')
            plt.ylabel('Intensity (arb. unit)')

        else:
            plt.plot(bin_centers, count, linestyle='-', color='b')
            plt.ylabel('Count')
        plt.xlabel('Energy')
        plt.title(spectrumTitle)
        if logarithmic:
            plt.yscale('log')
        plt.grid(True)
        plt.show()

    @staticmethod
    def lists_energy(list_countij, braggEngine_init):
        energyList = []

        for countij in list_countij:
            count = countij[0]
            iIndex = countij[1]
            jIndex = countij[2]

            yPixel = iIndex
            xPixel = jIndex

            x_0 = - braggEngine_init.xWidth / 2
            y_0 = + braggEngine_init.yWidth / 2

            x_coord = xPixel * braggEngine_init.pixelWidth + x_0  # x_o is such that the x coord for the exact center would be 0
            y_coord = y_0 - yPixel * braggEngine_init.pixelWidth

            energyVal = braggEngine_init.xyImagePlane_to_energy(x_coord, y_coord)

            energyList.extend([energyVal] * count)

        return energyList


if __name__ == "__main__":
    crysPitch = -0.3444672207603088
    CrysRoll = 0.018114148603524255
    CamPitch = 0.7950530342947064
    CamRoll = -0.005323879756451509
    rcam = 0.08395021
    thetacam = 2.567
    rcamSpherical = np.array([rcam, thetacam, np.pi])

    spectrum = Spectrum(8, crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical, removeTopRows=0,
                        how_many_sigma=2, noPhoton_adu_thr=30
                        )
    spectrum.multiPixelSpectrum(band_width=1, plotSpectrum=True, logarithmic=False, plotEachSubSpectrum=True,
                                intensity_arb_unit=True,
                                spectrumTitle=None,
                                intermediary_matrices_arg_dict={
                                    "folderpath": r"C:/Users/marcg/OneDrive/Documents/Oxford Physics/Year 3/B8/b8_xspeds/data_logs\image_matrices\image_8",
                                    "filename": "im8_test1"
                                })

    # spectrum = Spectrum(imData[8], 100, crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical)
    # spectrum.singlePixelPhotonSpectrumOLD(band_width=1, plotSpectrum=True, intensity_arbUnits=True, logarithmic=False)
    # spectrum.singlePixelPhotonSpectrum(band_width=1,plotSpectrum=True,title="Photon Energy Spectrum with Multi Photon Single Pixel Hits Model 1"
    #                                    ,logarithmic=False)
    # spectrum.simpleSpectrum(band_width=2,plotSpectrum=True,logarithmic=False)

    pass
