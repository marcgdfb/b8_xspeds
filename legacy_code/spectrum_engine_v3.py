from calibrate_geometry_v4 import *
from spc_engine_v4 import *
import time


# TODO: Make it clear in documentation that the mean is subtracted from the result. Do this in readme?

# TODO: Make sure that each image uses the fitted curves specific to that curve to correct for xray jitter

# TODO: Consider poisson error in this ~ sqrt(N)


class Spectrum:
    def __init__(self, indexOfInterest,
                 crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_camera_spherical,
                 sp_adu_thr=180, dp_adu_thr=240, noPhoton_adu_thr=50,
                 tp_adu_thr=400, quad_p_adu_thr=550, quint_p_adu_thr=700,
                 removeTopRows=0,
                 how_many_sigma=0,
                 folderpath="stored_variables"):

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
        self.tp_thr = tp_adu_thr
        self.quad_thr = quad_p_adu_thr
        self.quint_thr = quint_p_adu_thr

        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical

        self.removeTopRows = removeTopRows
        self.how_many_sigma = how_many_sigma

    def multiPixelSpectrum_eff_island(self, band_width=1,
                                      spectrumTitle=None,
                                      plotSpectrum=False, logarithmic=False,
                                      intensity_arb_unit=False,
                                      intermediary_matrices_arg_dict=None):

        if spectrumTitle is None:
            howManySigmaTitle = f'\nConsidering values {self.how_many_sigma} above the mean'
            thresholds_title1 = f'\nNo Photons < {self.noP_thresh},SP < {self.sp_thresh},DP < {self.dp_thresh}'
            thresholds_title2 = f'\nTP < {self.tp_thr},QuadP < {self.quad_thr},QuintP < {self.quint_thr}'
            spectrumTitle = f"Photon Energy Spectrum with Multi Photon Hits Version 2" + howManySigmaTitle + thresholds_title1 + thresholds_title2

        spc_engine = PhotonCounting(indexOfInterest=self.indexOfInterest,
                                    no_photon_adu_thr=self.noP_thresh, sp_adu_thr=self.sp_thresh,
                                    dp_adu_thr=self.dp_thresh, tp_adu_thr=self.tp_thr, quad_p_adu_thr=self.quad_thr,
                                    quint_p_adu_thr=self.quint_thr,
                                    removeRows0To_=self.removeTopRows,
                                    howManySigma_thr=self.how_many_sigma, )
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

        output_dictionaries, imMatPRemoved = spc_engine.efficient_imageScan(report=True,
                                                                            intermediary_matrices_arg_dict=intermediary_matrices_arg_dict)
        energy_dict = {}

        for key in output_dictionaries.keys():
            output_dict = output_dictionaries[key]
            listCountij = output_dict["list_countij"]
            energyList = self.lists_energy(listCountij, braggEngine_init=bragg_engine)
            energy_dict[key] = energyList

        energyList = []
        for key in output_dictionaries.keys():
            energyList.extend(energy_dict[key])

        if plotSpectrum:
            self.plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit, logarithmic)

        # The first 3 columns have straight lines going down the vertical
        imMatPRemoved[:, 0:3] = 0
        # imMatPRemoved[0:200, :] = 0

        results_island_dict = spc_engine.operateOnIslands(imMatPRemoved)
        island_list_Cij = results_island_dict["list_countij"]
        energyList_islands = self.lists_energy(island_list_Cij, braggEngine_init=bragg_engine)

        energyList.extend(energyList_islands)

        if plotSpectrum:
            self.plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit, logarithmic)

        return energyList

    def multiPixel_island(self, band_width=1,
                          spectrumTitle=None,
                          plotSpectrum=False, logarithmic=False,
                          intensity_arb_unit=False,
                          row_separate=0):

        if spectrumTitle is None:
            howManySigmaTitle = f'\nConsidering values {self.how_many_sigma} above the mean'
            thresholds_title1 = f'\nNo Photons < {self.noP_thresh},SP < {self.sp_thresh},DP < {self.dp_thresh}'
            thresholds_title2 = f'\nTP < {self.tp_thr},QuadP < {self.quad_thr},QuintP < {self.quint_thr}'
            spectrumTitle = f"Photon Energy Spectrum for image {self.indexOfInterest}" + howManySigmaTitle + thresholds_title1 + thresholds_title2

        spc_engine = PhotonCounting(indexOfInterest=self.indexOfInterest,
                                    no_photon_adu_thr=self.noP_thresh, sp_adu_thr=self.sp_thresh,
                                    dp_adu_thr=self.dp_thresh, tp_adu_thr=self.tp_thr, quad_p_adu_thr=self.quad_thr,
                                    quint_p_adu_thr=self.quint_thr,
                                    removeRows0To_=self.removeTopRows,
                                    howManySigma_thr=self.how_many_sigma, )
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

        moI = spc_engine.imMat

        # The first 3 columns have vertical lines that are due to edge effects
        moI[:, 0:3] = 0

        if row_separate != 0:
            moI_Top = moI[0:row_separate, :]
            moI_Top_above2sigma = np.where(moI_Top > 2 * spc_engine.sigmaPedestal, moI_Top, 0)

            moI_bottom = moI[row_separate:, :]

            results_island_dict_top = spc_engine.operateOnIslands(moI_Top_above2sigma)
            island_list_Cij_top = results_island_dict_top["list_countij"]
            energyList_islands_top = self.lists_energy(island_list_Cij_top, braggEngine_init=bragg_engine)

            results_island_dict_bot = spc_engine.operateOnIslands(moI_bottom)
            island_list_Cij_bot = results_island_dict_bot["list_countij"]
            energyList_islands = self.lists_energy(island_list_Cij_bot, braggEngine_init=bragg_engine)

            energyList_islands.extend(energyList_islands_top)
        else:
            results_island_dict = spc_engine.operateOnIslands(moI)
            island_list_Cij = results_island_dict["list_countij"]
            energyList_islands = self.lists_energy(island_list_Cij, braggEngine_init=bragg_engine)

        if plotSpectrum:
            self.plotSpectrum(energyList_islands, band_width, spectrumTitle, intensity_arb_unit, logarithmic)

        return energyList_islands, moI

    @staticmethod
    def plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit=False, logarithmic=False):

        # energyBins = np.arange(min(energyList), max(energyList) + band_width, band_width)
        energyBins = np.arange(1000, 1700 + band_width, band_width)

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

            energyVal = braggEngine_init.xyPixelImagePlane_to_energy(xPixel_imPlane=xPixel, yPixel_imPlane=yPixel)

            energyList.extend([energyVal] * count)

        return energyList


def dict_ij_perEnergyBin(band_width=1):
    print("-" * 30)
    print("Creating dictionary of ij coordinates in each bin")

    energy_of_pixelMat = np.load(
        r"../old_logs_and_stored_variables/v2/energy_of_pixel.npy")

    energyBands = np.arange(1000, 1700 + band_width, band_width)

    dict_bin_indices = {}
    for i in range(len(energyBands) - 1):
        dict_bin_indices[(int(energyBands[i]), int(energyBands[i + 1]))] = []

    # print(dict_bin_indices.keys())

    # Iterate over matrix indices
    for i in range(energy_of_pixelMat.shape[0]):
        for j in range(energy_of_pixelMat.shape[1]):
            energy_value = energy_of_pixelMat[i, j]

            # Find the correct bin. The -1 is to ensure that it counts from 0
            bin_index = np.searchsorted(energyBands, energy_value, side='right') - 1

            # Ensure it's within the valid bin range
            if 0 <= bin_index < len(energyBands) - 1:
                bin_range = (int(energyBands[bin_index]), int(energyBands[bin_index + 1]))
                dict_bin_indices[bin_range].append((i, j))

    return dict_bin_indices


def determine_solidAnglePerEnergyBin(band_width=1, numberOfPoints=1, plotdistribtion=False, saveToExcel=False):
    dict_ij_perBin = dict_ij_perEnergyBin(band_width=band_width)

    print("-" * 30)
    print("Creating dictionary of total Solid Angle captured for each bin")
    print("Start:", time.strftime("%H:%M:%S", time.localtime()))

    solidAngle_engine = SolidAngle()

    totalSolidAngle_dict = {}

    for key in dict_ij_perBin.keys():
        print("key: ", key)
        list_ij = dict_ij_perBin[key]

        totalSolidAngle = 0

        for ij_idices in list_ij:
            i_idx = ij_idices[0]
            j_idx = ij_idices[1]

            totalSolidAngle += solidAngle_engine.solidAngle_pixelij(i_idx, j_idx, number_points=numberOfPoints)

        print(totalSolidAngle)
        totalSolidAngle_dict[key] = totalSolidAngle

    print("Finish:", time.strftime("%H:%M:%S", time.localtime()))

    if plotdistribtion:
        bin_bounds = np.array(list(totalSolidAngle_dict.keys()))  # Convert dict keys (tuples) to array
        bin_centers = (bin_bounds[:, 0] + bin_bounds[:, 1]) / 2  # Compute bin centers
        total_solid_angles = np.array(list(totalSolidAngle_dict.values()))  # Extract solid angles

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, total_solid_angles, marker='o', linestyle='-', color='b', label="Total Solid Angle")
        plt.xlabel("Energy Bin Center (eV)")
        plt.ylabel("Total Solid Angle")
        plt.title("Distribution of Solid Angle")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    if saveToExcel:

        l_lb = []
        l_ub = []
        l_solidAngle = []

        for key in totalSolidAngle_dict.keys():
            l_lb.append(key[0])
            l_ub.append(key[1])
            l_solidAngle.append(totalSolidAngle_dict[key])

        df = pd.DataFrame()
        df["Lower Bound"] = l_lb
        df["Upper Bound"] = l_ub
        df["Solid Angle"] = l_solidAngle

        excl_filename = r"../old_logs_and_stored_variables/v2/solidAngle.xlsx"

        df.to_excel(excl_filename, index=False)

    return totalSolidAngle_dict


def averageAllImages(intensity_arb_unit=False, band_width=1,
                     howManySigma=2):
    energyList_total = []

    thr_np = 80
    thr_sp = 150
    thr_dp = 250
    thr_tp = 400
    thr_quadp = 550
    thr_quintp = 700


    energy_lists_dict = {}
    theta_cam = 2.567

    for im_number in range(len(loadData())):

        if im_number in [0, 3, 5, 9, 10, 12, 13, 15, 18]:
            continue

        geo_engine = geo_engine_withSavedParams(im_number)

        cryPitch = geo_engine.crystal_pitch
        CryRoll = geo_engine.crystal_roll
        Cam_Pitch = geo_engine.camera_pitch
        Cam_Roll = geo_engine.camera_roll
        r_cam = geo_engine.r_cam

        # assume initially that all have the same crystal orientation:

        spectrum_engine = Spectrum(indexOfInterest=im_number,
                                   crystal_pitch=cryPitch, crystal_roll=CryRoll,
                                   camera_pitch=Cam_Pitch, camera_roll=Cam_Roll,
                                   r_camera_spherical=np.array([r_cam, theta_cam, np.pi]),
                                   removeTopRows=0,
                                   how_many_sigma=howManySigma, noPhoton_adu_thr=thr_np, sp_adu_thr=thr_sp,
                                   dp_adu_thr=thr_dp,
                                   tp_adu_thr=thr_tp,
                                   quad_p_adu_thr=thr_quadp,
                                   quint_p_adu_thr=thr_quintp,
                                   )

        energyList_islands, moI = spectrum_engine.multiPixel_island(band_width=band_width, plotSpectrum=False)

        energy_lists_dict[im_number] = energyList_islands

        energyList_total.extend(energyList_islands)

    howManySigmaTitle = f'\nConsidering values {howManySigma} above the mean'
    thresholds_title1 = f'\nNo Photons < {thr_np},SP < {thr_sp},DP < {thr_dp}'
    thresholds_title2 = f'\nTP < {thr_tp},QuadP < {thr_quadp},QuintP < {thr_quintp}'
    spectrumTitle = f"Total Photon Energy Spectrum with Multi Photon Hits Version 2" + howManySigmaTitle + thresholds_title1 + thresholds_title2

    Spectrum(0, 0, 0, 0, 0, np.array([0, 2.567, np.pi])).plotSpectrum(
        energyList_total, band_width=band_width,
        spectrumTitle=spectrumTitle,
        intensity_arb_unit=intensity_arb_unit)


if __name__ == "__main__":
    def check_spec():
        crysPitch = -0.3444672207603088
        CrysRoll = 0.018114148603524255
        CamPitch = 0.7950530342947064
        CamRoll = -0.005323879756451509
        rcam = 0.08395021
        thetacam = 2.567
        rcamSpherical = np.array([rcam, thetacam, np.pi])

        spectrum = Spectrum(8, crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical, removeTopRows=0,
                            how_many_sigma=2, noPhoton_adu_thr=80, sp_adu_thr=150, dp_adu_thr=250, tp_adu_thr=400,
                            quad_p_adu_thr=550,
                            quint_p_adu_thr=700
                            )
        # spectrum.multiPixelSpectrum(band_width=1, plotSpectrum=True, logarithmic=False, plotEachSubSpectrum=True,
        #                             intensity_arb_unit=True,
        #                             spectrumTitle=None,
        #                             intermediary_matrices_arg_dict={
        #                                 "folderpath": r"C:/Users/marcg/OneDrive/Documents/Oxford Physics/Year 3/B8/b8_xspeds/data_logs\image_matrices\image_8",
        #                                 "filename": "im8_test1"
        #                             })

        _, imMat = spectrum.multiPixel_island(band_width=1, plotSpectrum=True, logarithmic=False,
                                              intensity_arb_unit=False)

        plt.imshow(imMat)
        plt.show()


    # check_spec()

    # averageAllImages(intensity_arb_unit=True, band_width=2,howManySigma=2)

    determine_solidAnglePerEnergyBin(1, 3, True,saveToExcel=True)

    pass
