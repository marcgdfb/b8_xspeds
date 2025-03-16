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
                 how_many_sigma=2,
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
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        print(f"count: length = {len(count)}")
        print(f"bin_centers: length = {len(bin_centers)}")

        index_folder = os.path.join(self.folderpath, str(self.indexOfInterest))
        solidAng_filepath = os.path.join(index_folder, f"solid_angle_of_binwidth_{bin_width}.xlsx")
        try:
            df_solid_angle = pd.read_excel(solidAng_filepath)
        except FileNotFoundError:
            print(f"File not found: {solidAng_filepath}")
            df_solid_angle = pd.DataFrame(solid_angle_per_energy_bin(index_of_interest=self.indexOfInterest, bin_width=bin_width))

        df_solid_angle = df_solid_angle[['bins','solid_angle']]

        # Convert string representations of tuples to actual tuples
        df_solid_angle["bins"] = df_solid_angle["bins"].apply(ast.literal_eval)

        solidAngle_bins_tuple = df_solid_angle['bins'].values
        solid_angle_vals = df_solid_angle['solid_angle'].values

        solidAngle_bins_center_dict = {}
        for tuple_sa,solid_angle_val in zip(solidAngle_bins_tuple,solid_angle_vals):
            center_val = (tuple_sa[0] + tuple_sa[1])/2
            solidAngle_bins_center_dict[center_val] = solid_angle_val

        solid_angle_normalised_counts = []
        solid_angle_normalised_uncertainty = []
        count_uncertainty = []
        for count, count_center in zip(count,bin_centers):
            if count == 0:
                solid_angle_normalised_counts.append(0)
                count_uncertainty.append(0)
                solid_angle_normalised_uncertainty.append(0)
                continue

            # Solid Angle normalisation of count
            normalised_count = count / solidAngle_bins_center_dict[count_center]
            solid_angle_normalised_counts.append(normalised_count)

            # Poisson Uncertainty
            uncertainty = np.sqrt(count)
            count_uncertainty.append(uncertainty)
            # Solid Angle Normalisation of uncertainty
            normalised_uncertainty = uncertainty / solidAngle_bins_center_dict[count_center]
            solid_angle_normalised_uncertainty.append(normalised_uncertainty)


        normalised_counts = np.array(solid_angle_normalised_counts)
        normalised_count_unc = np.array(solid_angle_normalised_uncertainty)

        count_unc = np.array(count_uncertainty)

        if save:
            count_filepath = os.path.join(index_folder, "count.npy")
            np.save(count_filepath, count)

            count_unc_filepath = os.path.join(index_folder, "count_unc.npy")
            np.save(count_unc_filepath, count_unc)

            normalised_counts_filepath = os.path.join(index_folder, "solidAng_normalised_counts.npy")
            np.save(normalised_counts_filepath, normalised_counts)

            normalised_counts_unc_filepath = os.path.join(index_folder, "solidAng_normalised_counts_unc.npy")
            np.save(normalised_counts_unc_filepath, normalised_counts)

            bin_center_filepath = os.path.join(index_folder, "bin_centers.npy")
            np.save(bin_center_filepath, bin_centers)

        return count, count_unc,normalised_counts, normalised_count_unc,bin_centers

    def plotSpectrum_solid_angle(self,bin_width=1,intensity_arb_unit=False,logarithmic=False,plotUnc=True):
        count, count_unc,normalised_counts, normalised_count_unc,bin_centers = self.islandSpectrum_SolidAngle_with_uncertainty(bin_width=bin_width)

        plt.figure(figsize=(10, 6))
        if intensity_arb_unit:
            countIntensity = []
            countIntensity_lb = []
            countIntensity_ub = []
            for countE, count_unc,center in zip(normalised_counts,normalised_count_unc, bin_centers):

                intensity_ = countE * center
                intensity_uncertainty = count_unc * center

                countIntensity.append(intensity_)
                countIntensity_lb.append(intensity_ - intensity_uncertainty)
                countIntensity_ub.append(intensity_ + intensity_uncertainty)

            countIntensity = np.array(countIntensity)
            countIntensity_ub = np.array(countIntensity_ub)
            countIntensity_lb = np.array(countIntensity_lb)

            if plotUnc:
                plt.plot(bin_centers, countIntensity_lb, linestyle='-', color='r')
                plt.plot(bin_centers, countIntensity_ub, linestyle='-', color='r')

            plt.plot(bin_centers, countIntensity, linestyle='-', color='b')
            plt.ylabel('Intensity (arb. unit)')

        else:
            if plotUnc:
                count_ub = normalised_counts + normalised_count_unc
                count_lb = normalised_counts - normalised_count_unc

                plt.plot(bin_centers, count_lb, linestyle='-', color='r')
                plt.plot(bin_centers, count_ub, linestyle='-', color='r')

            plt.plot(bin_centers, normalised_counts, linestyle='-', color='b')
            plt.ylabel('Count')
        plt.xlabel('Energy')
        plt.title("")
        if logarithmic:
            plt.yscale('log')
        plt.grid(True)
        plt.show()


    @staticmethod
    def plotSpectrum_count_unc_binCenters(count_array,unc_array,bin_center_array,intensity_arb_unit=False,logarithmic=False,plotUnc=True):

        count_lb = count_array - unc_array
        count_ub = count_array + unc_array

        plt.figure(figsize=(10, 6))
        if intensity_arb_unit:
            countIntensity = []
            countIntensity_lb = []
            countIntensity_ub = []
            for COUNT, COUNT_LB, COUNT_UB, center in zip(count_array,count_lb, count_ub, bin_center_array):
                countIntensity.append(COUNT * center)
                countIntensity_lb.append(COUNT_LB * center)
                countIntensity_ub.append(COUNT_UB * center)

            countIntensity = np.array(countIntensity)
            countIntensity_ub = np.array(countIntensity_ub)
            countIntensity_lb = np.array(countIntensity_lb)

            if plotUnc:
                plt.plot(bin_center_array, countIntensity_lb, linestyle='-', color='r')
                plt.plot(bin_center_array, countIntensity_ub, linestyle='-', color='r')

            plt.plot(bin_center_array, countIntensity, linestyle='-', color='b')
            plt.ylabel('Intensity (arb. unit)')

        else:
            if plotUnc:
                plt.plot(bin_center_array, count_lb, linestyle='-', color='r')
                plt.plot(bin_center_array, count_ub, linestyle='-', color='r')

            plt.plot(bin_center_array, count_array, linestyle='-', color='b')
            plt.ylabel('Count')
        plt.xlabel('Energy')
        plt.title("")
        if logarithmic:
            plt.yscale('log')
        plt.grid(True)
        plt.show()


# TODO: make a graph with both solid angle corrected and non solid angle corrected!



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

def collect_count_unc(bin_width=1, folderpath="stored_variables",save=True):

    energyBins = np.arange(1000, 1700 + bin_width, bin_width)
    bin_centers = (energyBins[:-1] + energyBins[1:]) / 2

    dict_counts = {}

    for center in bin_centers:
        dict_counts[center] = {"count": 0,
                               "count_unc": 0}

    try:
        list_folderpath = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

        for folder_path_ in list_folderpath:
            print(folder_path_)
            if os.path.exists(os.path.join(folderpath, "solidAng_normalised_counts.npy")):

                counts = np.load(os.path.join(folderpath, "solidAng_normalised_counts.npy"))
                unc_counts = np.load(os.path.join(folderpath, "solidAng_normalised_counts_unc.npy"))
                bin_centers_ = np.load(os.path.join(folderpath, "bin_centers.npy"))

                for count_,unc_,center_ in zip(counts,unc_counts,bin_centers_):

                    dict_center_ = dict_counts[center_]
                    dict_center_["count"] += count_
                    dict_center_["count_unc"] += unc_

            else:
                print("Count Files not saved ")

        count_list = []
        count_unc_list = []
        for bin_center in bin_centers:
            count_list.append(dict_counts[bin_center]["count"])
            count_unc_list.append(dict_counts[bin_center]["count_unc"])

        count_array = np.array(count_list)
        count_unc_array = np.array(count_unc_list)

        if save:
            filename_count = os.path.join(folderpath, "solidAng_normalised_counts_total.npy")
            filename_unc = os.path.join(folderpath,"solidAng_normalised_counts_unc_tot.npy")
            filename_bins = os.path.join(folderpath, "solidAng_normalised_counts_bins.npy")

            np.save(filename_count, count_array)
            np.save(filename_unc, count_unc_array)
            np.save(filename_bins, bin_centers)


    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")




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


# TODO: save Bin center, count , count_unc


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
        spectrum_eng = Spectrum(indexOfInterest=indexOI, folderpath="stored_variables",how_many_sigma=2,)
        # spectrum_eng.islandSpectrum_SolidAngle_with_uncertainty(bin_width=1, save=True)
        spectrum_eng.plotSpectrum_solid_angle(bin_width=1,intensity_arb_unit=True,)

    solidAngle(8)


    # solid_angle_per_energy_bin(8,1,plotDistribtion=True,save=False)



    pass
