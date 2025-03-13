from calibrate_geometry_v4 import *
from spc_engine_v4 import *
import time
import pandas as pd
import ast


# TODO: Make it clear in documentation that the mean is subtracted from the result. Do this in readme?

# TODO: Make sure that each image uses the fitted curves specific to that curve to correct for xray jitter

# TODO: Consider poisson error in this ~ sqrt(N)


class Spectrum:
    def __init__(self, indexOfInterest,
                 geo_engine=None,
                 sp_adu_thr=160, dp_adu_thr=250, noPhoton_adu_thr=80,
                 tp_adu_thr=400, quad_p_adu_thr=550, quint_p_adu_thr=700,
                 removeTopRows=0,
                 how_many_sigma=0,
                 folderpath="stored_variables"):

        if geo_engine is None:
            geo_engine = geo_engine_withSavedParams(indexOfInterest)

        crystal_pitch = geo_engine.crystal_pitch
        crystal_roll = geo_engine.crystal_roll
        camera_pitch = geo_engine.camera_pitch
        camera_roll = geo_engine.camera_roll
        r_camera_spherical = geo_engine.r_camera_spherical


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

        self.folderpath = folderpath

    def multiPixelSpectrum_eff_island(self, bin_wdith=1,
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
            self.plotSpectrum(energyList, bin_wdith, spectrumTitle, intensity_arb_unit, logarithmic)

        # The first 3 columns have straight lines going down the vertical
        imMatPRemoved[:, 0:3] = 0
        # imMatPRemoved[0:200, :] = 0

        results_island_dict = spc_engine.operateOnIslands(imMatPRemoved)
        island_list_Cij = results_island_dict["list_countij"]
        energyList_islands = self.lists_energy(island_list_Cij, braggEngine_init=bragg_engine)

        energyList.extend(energyList_islands)

        if plotSpectrum:
            self.plotSpectrum(energyList, bin_wdith, spectrumTitle, intensity_arb_unit, logarithmic)

        return energyList

    def multiPixel_island(self, bin_wdith=1,
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
            self.plotSpectrum(energyList_islands, bin_wdith, spectrumTitle, intensity_arb_unit, logarithmic)

        return energyList_islands, moI

    @staticmethod
    def plotSpectrum(energyList, bin_wdith, spectrumTitle, intensity_arb_unit=False, logarithmic=False):

        # energyBins = np.arange(min(energyList), max(energyList) + bin_wdith, bin_wdith)
        energyBins = np.arange(1000, 1700 + bin_wdith, bin_wdith)

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


    def islandSpectrum_SolidAngle_with_uncertainty(self,bin_width=1,save=False):
        energyList_islands, _ = self.multiPixel_island(bin_wdith=bin_width, plotSpectrum=False)

        energyBins = np.arange(1000, 1700 + bin_width, bin_width)
        photonEnergies = np.array(energyList_islands)
        count, bin_edges = np.histogram(photonEnergies, energyBins)
        # Convert np.histogram edges into bin tuples
        count_bins = [f"({int(bin_edges[i])}, {int(bin_edges[i + 1])})" for i in range(len(bin_edges) - 1)]
        # Create a DataFrame for counts
        counts_df = pd.DataFrame({'bins': count_bins, 'counts': count})
        counts_df["count_uncertainty"] = np.sqrt(counts_df["counts"])


        index_folder = os.path.join(self.folderpath, str(self.indexOfInterest))
        solidAng_filepath = os.path.join(index_folder, f"solid_angle_of_binwidth_{bin_width}.xlsx")
        try:
            df_solid_angle = pd.read_excel(solidAng_filepath)
        except FileNotFoundError:
            print(f"File not found: {solidAng_filepath}")
            df_solid_angle = pd.DataFrame(solid_angle_per_energy_bin(index_of_interest=self.indexOfInterest, bin_width=bin_width))

        df_solid_angle = df_solid_angle[['bins','solid_angle']]
        merged_df = pd.merge(counts_df, df_solid_angle, on='bins', how='left')
        merged_df["normalised_counts"] = merged_df["counts"] / merged_df["solid_angle"]
        merged_df["normalised_count_uncertainty"] = merged_df["count_uncertainty"] / merged_df["solid_angle"]
        merged_df = merged_df[merged_df['counts'] != 0]

        if save:
            excelFilepath = os.path.join(index_folder, f"spectrum_data_binwidth_{bin_width}.xlsx")
            merged_df.to_excel(excelFilepath, index=False)

    def plotSpectrumDf(self,bin_width=1,intensity_arb_unit=False,logarithmic=False):
        index_folder = os.path.join(self.folderpath, str(self.indexOfInterest))
        excelFilepath = os.path.join(index_folder, f"spectrum_data_binwidth_{bin_width}.xlsx")
        df_spectrum = pd.read_excel(excelFilepath)

        oldCount = np.array(df_spectrum["normalised_counts"].values)

        df_spectrum["bins"] = df_spectrum["bins"].apply(ast.literal_eval)
        binTuples = df_spectrum["bins"].values
        bin_centers = np.array([(start + end) / 2 for start, end in binTuples])


        plt.figure(figsize=(10, 6))
        if intensity_arb_unit:
            countIntensity = []
            for countE, center in zip(oldCount, bin_centers):
                countIntensity.append(countE * center)
            countIntensity = np.array(countIntensity)
            plt.plot(bin_centers, countIntensity, linestyle='-', color='b')
            plt.ylabel('Intensity (arb. unit)')

        else:
            plt.plot(bin_centers, oldCount, linestyle='-', color='b')
            plt.ylabel('Count')
        plt.xlabel('Energy')
        plt.title("")
        if logarithmic:
            plt.yscale('log')
        plt.grid(True)
        plt.show()





def dict_ij_perEnergyBin(index_of_interest, bin_width=1, folderpath="stored_variables"):
    print("-" * 30)
    print("Creating dictionary of ij coordinates in each bin")

    index_folder = os.path.join(folderpath, str(index_of_interest))
    energy_filepath = os.path.join(index_folder, "energy_of_pixel.npy")
    energy_of_pixelMat = np.load(energy_filepath)

    energyBands = np.arange(1000, 1700 + bin_width, bin_width)

    dict_bin_indices = {}
    for i in range(len(energyBands) - 1):
        dict_bin_indices[(int(energyBands[i]), int(energyBands[i + 1]))] = []


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

def solid_angle_per_energy_bin(index_of_interest, bin_width=1, folderpath="stored_variables",
                               plotDistribtion=False, save=True):
    dict_ij_perBin = dict_ij_perEnergyBin(index_of_interest, bin_width=bin_width, folderpath=folderpath)

    print("-" * 30)
    print("Creating dictionary of total Solid Angle captured for each bin")
    print("Start:", time.strftime("%H:%M:%S", time.localtime()))

    index_folder = os.path.join(folderpath, str(index_of_interest))
    solidAngle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
    solidAng_mat = np.load(solidAngle_filepath)

    totalSolidAngle_dict = {}

    for key in dict_ij_perBin.keys():
        # print("key: ", key)
        list_ij = dict_ij_perBin[key]

        totalSolidAngle = 0

        for ij_idices in list_ij:
            i_idx = ij_idices[0]
            j_idx = ij_idices[1]

            totalSolidAngle += solidAng_mat[i_idx, j_idx]

        # print(totalSolidAngle)
        totalSolidAngle_dict[key] = totalSolidAngle

    print("Finish:", time.strftime("%H:%M:%S", time.localtime()))

    solid_angle_df = pd.DataFrame(totalSolidAngle_dict.items(), columns=['bins', 'solid_angle'])
    solid_angle_df = solid_angle_df[solid_angle_df['solid_angle'] != 0]
    if save:
        bin_solid_angle_filepath = os.path.join(index_folder, f"solid_angle_of_binwidth_{bin_width}.xlsx")
        solid_angle_df.to_excel(bin_solid_angle_filepath, index=False)

    if plotDistribtion:
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

    return solid_angle_df

def averageAllImages(intensity_arb_unit=False, bin_wdith=1,
                     howManySigma=2,folderpath="stored_variables"):

    energyList_total = []

    thr_np = 80
    thr_sp = 150
    thr_dp = 250
    thr_tp = 400
    thr_quadp = 550
    thr_quintp = 700


    energy_lists_dict = {}

    energyBins = np.arange(1000, 1700 + bin_wdith, bin_wdith)

    for im_number in range(len(loadData())):

        if im_number in list_data:
            continue

        geo_engine = geo_engine_withSavedParams(im_number)


        spectrum_engine = Spectrum(indexOfInterest=im_number,
                                   geo_engine=geo_engine,
                                   removeTopRows=0,
                                   how_many_sigma=howManySigma, noPhoton_adu_thr=thr_np, sp_adu_thr=thr_sp,
                                   dp_adu_thr=thr_dp,
                                   tp_adu_thr=thr_tp,
                                   quad_p_adu_thr=thr_quadp,
                                   quint_p_adu_thr=thr_quintp,
                                   )

        energyList_islands, moI = spectrum_engine.multiPixel_island(bin_wdith=bin_wdith, plotSpectrum=False)

        energy_lists_dict[im_number] = energyList_islands

        energyList_total.extend(energyList_islands)

        photonEnergies = np.array(energyList_islands)
        count, bins_edges = np.histogram(photonEnergies, energyBins)
        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        index_folder = os.path.join(folderpath, str(im_number))
        solidAng_filepath = os.path.join(index_folder, f"solid_angle_of_binwidth_{bin_wdith}.xlsx")

        

    howManySigmaTitle = f'\nConsidering values {howManySigma} above the mean'
    thresholds_title1 = f'\nNo Photons < {thr_np},SP < {thr_sp},DP < {thr_dp}'
    thresholds_title2 = f'\nTP < {thr_tp},QuadP < {thr_quadp},QuintP < {thr_quintp}'
    spectrumTitle = f"Total Photon Energy Spectrum with Multi Photon Hits Version 2" + howManySigmaTitle + thresholds_title1 + thresholds_title2

    Spectrum(0, ).plotSpectrum(
        energyList_total, bin_wdith=bin_wdith,
        spectrumTitle=spectrumTitle,
        intensity_arb_unit=intensity_arb_unit)


if __name__ == "__main__":
    def check_spec(indexOI):
        spectrum = Spectrum(indexOI, removeTopRows=0,
                            how_many_sigma=2, noPhoton_adu_thr=80, sp_adu_thr=150, dp_adu_thr=250, tp_adu_thr=400,
                            quad_p_adu_thr=550,
                            quint_p_adu_thr=700
                            )
        # spectrum.multiPixelSpectrum(bin_wdith=1, plotSpectrum=True, logarithmic=False, plotEachSubSpectrum=True,
        #                             intensity_arb_unit=True,
        #                             spectrumTitle=None,
        #                             intermediary_matrices_arg_dict={
        #                                 "folderpath": r"C:/Users/marcg/OneDrive/Documents/Oxford Physics/Year 3/B8/b8_xspeds/data_logs\image_matrices\image_8",
        #                                 "filename": "im8_test1"
        #                             })

        _, imMat = spectrum.multiPixel_island(bin_wdith=1, plotSpectrum=True, logarithmic=False,
                                              intensity_arb_unit=False)




    # check_spec(11)

    # averageAllImages(intensity_arb_unit=True, bin_wdith=2,howManySigma=2)


    def solidAngle(indexOI=11):
        spectrum_eng = Spectrum(indexOfInterest=indexOI, folderpath="stored_variables")
        spectrum_eng.islandSpectrum_SolidAngle_with_uncertainty(bin_width=1, save=True)
        spectrum_eng.plotSpectrumDf(bin_width=1)

    solidAngle(11)

    pass
