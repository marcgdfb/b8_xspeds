import matplotlib.pyplot as plt

from calibrate_geometry_v4 import *
from spc_engine_5 import *
import time
import pandas as pd
import ast

# TODO: save each individual spectrum count, uncertainty, SOlidAngle normalised count, uncertainty,


class Spectrum:
    def __init__(self, indexOfInterest,
                 geo_engine=None,
                 removeTopRows=0,
                 how_many_sigma=1,
                 folderpath="stored_variables",
                 spectrumTitle=None):

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

            if removeTopRows > 0:
                print("removeTopRows = ", removeTopRows)
            print("how_many_sigma = ", how_many_sigma)

        printVar()
        self.indexOfInterest = indexOfInterest

        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical

        self.bragg_eng = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

        self.removeTopRows = removeTopRows
        self.how_many_sigma = how_many_sigma

        self.folderpath = folderpath

        if spectrumTitle is None:
            howManySigmaTitle = f'\nConsidering values {self.how_many_sigma} above the mean'
            # thresholds_title1 = f'\nAcceptance Region: {self.noP_thresh} < ADU total < {self.adu_cap}'
            # thresholds_title2 = f'\nSingle Photon ADU = {self.sp_thresh} with allowed offset = {self.adu_offset}'
            spectrumTitle = f"Photon Energy Spectrum for image {self.indexOfInterest}" + howManySigmaTitle
                             # + thresholds_title1 + thresholds_title2)

        self.spectrumTitle = spectrumTitle


    def energy_list_set_shapes(self, preset_shape_adu_lowerBound=100,save=True):

        preset_eng = Preset_Shape_PhotonCounting(self.indexOfInterest,how_many_sigma=self.how_many_sigma,)
        dict_arrays_adu_ij = preset_eng.access_saved_adu_ij()

        dict_energy_lists = {}

        for key in dict_arrays_adu_ij.keys():
            array_adu_ij = dict_arrays_adu_ij[key]

            count_ij = []

            for row in array_adu_ij:
                adu, i_idx, j_idx = row

                if adu > preset_shape_adu_lowerBound:
                    count_ij.append([1,i_idx,j_idx])

            energy_list_key = self.lists_energy(list_countij=count_ij,braggEngine_init=self.bragg_eng)

            dict_energy_lists[key] = energy_list_key

            if save:
                sigma_spec_folderpath = self.sigma_thr_spectrum_folderpath()
                filename = f"energy_list_{key}.npy"
                np.save(os.path.join(sigma_spec_folderpath, filename), np.array(energy_list_key))

        return dict_energy_lists

    def energy_list_island(self,save=True,save_filename=f"energy_list_island.npy"):

        preset_eng = Preset_Shape_PhotonCounting(self.indexOfInterest,how_many_sigma=self.how_many_sigma,)
        arr_count_ij_island = preset_eng.access_saved_island_countij()

        energy_list_island = self.lists_energy(list_countij=arr_count_ij_island,braggEngine_init=self.bragg_eng)

        if save:
            sigma_spec_folderpath = self.sigma_thr_spectrum_folderpath()
            filename = save_filename
            np.save(os.path.join(sigma_spec_folderpath, filename), np.array(energy_list_island))

        return np.array(energy_list_island)

    def spectrum_set_shapes(self,bin_width=1):
        preset_eng = Preset_Shape_PhotonCounting(self.indexOfInterest, how_many_sigma=self.how_many_sigma, )
        unique_keys_ = preset_eng.unique_keys

        energyBins = np.arange(1000, 1000 + (700 // bin_width + 1) * bin_width, bin_width)

        try:
            dict_energy_vals = {}
            for key in unique_keys_:
                sigma_spec_folderpath = self.sigma_thr_spectrum_folderpath()
                filename = f"energy_list_{key}.npy"
                arr_energy_list = np.load(os.path.join(sigma_spec_folderpath, filename))
                dict_energy_vals[key] = arr_energy_list

        except FileNotFoundError as e:
            print("File not found: {}".format(e))
            print("Running energy_list_set_shapes")

            dict_energy_vals = self.energy_list_set_shapes()


        title_set_shapes = self.spectrumTitle + f"\nUsing only preset shapes (No Island)"
        self.plotSpectrum_fromDict(dictionary_energy_vals=dict_energy_vals,
                                   bin_edges=energyBins,
                                   title=title_set_shapes,
                                   ylabel="Count")


    def spectrum_set_then_island(self, bin_width=1,
                                 island_filename=f"energy_list_island.npy"):
        preset_eng = Preset_Shape_PhotonCounting(self.indexOfInterest, how_many_sigma=self.how_many_sigma, )
        unique_keys_ = preset_eng.unique_keys

        energyBins = np.arange(1000, 1000 + (700 // bin_width + 1) * bin_width, bin_width)

        try:
            dict_energy_vals = {}
            for key in unique_keys_:
                sigma_spec_folderpath = self.sigma_thr_spectrum_folderpath()
                filename = f"energy_list_{key}.npy"
                arr_energy_list = np.load(os.path.join(sigma_spec_folderpath, filename))
                dict_energy_vals[key] = arr_energy_list

        except FileNotFoundError as e:
            print("File not found: {}".format(e))
            print("Running energy_list_set_shapes")

            dict_energy_vals = self.energy_list_set_shapes()

        try:
            sigma_spec_folderpath = self.sigma_thr_spectrum_folderpath()
            filename = island_filename

            energy_list_island = np.load(os.path.join(sigma_spec_folderpath, filename))

            dict_energy_vals["Island"] = energy_list_island
        except FileNotFoundError as e:
            print("File not found: {}".format(e))
            print("Running energy_list_set_shapes")

            e_list_island = self.energy_list_island()
            dict_energy_vals["Island"] = e_list_island

        title_set_shapes = self.spectrumTitle + f"\nUsing all shapes with island maxing out at 2 photons"
        self.plotSpectrum_fromDict(dictionary_energy_vals=dict_energy_vals,
                                   bin_edges=energyBins,
                                   title=title_set_shapes,
                                   ylabel="Count")


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


    @staticmethod
    def hist_vals_bin_centers(energy_list, bin_width):

        energyBins = np.arange(1000, 1000+(700 // bin_width + 1) * bin_width, bin_width)

        photonEnergies = np.array(energy_list)
        hist_vals, bins_edges = np.histogram(photonEnergies, energyBins)
        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        return hist_vals, bin_centers

    @staticmethod
    def hist_vals_bin_edges(energy_list,bin_edges):

        photonEnergies = np.array(energy_list)
        hist_vals, bins_edges = np.histogram(photonEnergies, bin_edges)
        return hist_vals, bin_edges

    @staticmethod
    def energy_multiplied_hist_vals(hist_vals, bin_centers):
        count_intensity = []
        for countE, center in zip(hist_vals, bin_centers):
            count_intensity.append(countE * center)

        arr_count_intensity = np.load(count_intensity)

        return arr_count_intensity,bin_centers


    def spectrum_folderpath(self):
        index_folder = os.path.join(self.folderpath, str(self.indexOfInterest))
        spectrum_folderpath = os.path.join(index_folder, "spectrum")
        if not os.path.exists(spectrum_folderpath):
            os.makedirs(spectrum_folderpath)
        return spectrum_folderpath

    def sigma_thr_spectrum_folderpath(self):
        spectrum_folderpath = self.spectrum_folderpath()
        sigma_folderpath = os.path.join(spectrum_folderpath, f"{self.how_many_sigma}_sigma")
        if not os.path.exists(sigma_folderpath):
            os.makedirs(sigma_folderpath)
        return sigma_folderpath

    @staticmethod
    def plotSpectrum_fromDict(dictionary_energy_vals,bin_edges,title,ylabel):
        # Plot stacked histogram
        plt.figure(figsize=(10, 5))
        plt.hist(dictionary_energy_vals.values(), bins=bin_edges, stacked=True, label=dictionary_energy_vals.keys(), alpha=0.7,
                 edgecolor="black")

        plt.xlabel("ADU")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()





class Spectrum_old:
    def __init__(self, indexOfInterest,
                 geo_engine=None,
                 no_photon_adu_thr=80, sp_adu_thr=150, adu_offset=40, adu_cap=1600,
                 removeTopRows=0,
                 how_many_sigma=2,
                 folderpath="stored_variables",
                 spectrumTitle=None):

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

            print("noPhoton_adu_thr = ", no_photon_adu_thr)
            print("adu cap = ", adu_cap)
            print("sp_adu_thr = ", sp_adu_thr)
            print("adu offset = ", adu_offset)
            print("removeTopRows = ", removeTopRows)
            print("how_many_sigma = ", how_many_sigma)

        printVar()
        self.indexOfInterest = indexOfInterest
        self.noP_thresh = no_photon_adu_thr
        self.adu_cap = adu_cap
        self.sp_thresh = sp_adu_thr
        self.adu_offset = adu_offset

        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical

        self.removeTopRows = removeTopRows
        self.how_many_sigma = how_many_sigma

        self.folderpath = folderpath

        if spectrumTitle is None:
            howManySigmaTitle = f'\nConsidering values {self.how_many_sigma} above the mean'
            thresholds_title1 = f'\nAcceptance Region: {self.noP_thresh} < ADU total < {self.adu_cap}'
            thresholds_title2 = f'\nSingle Photon ADU = {self.sp_thresh} with allowed offset = {self.adu_offset}'
            spectrumTitle = f"Photon Energy Spectrum for image {self.indexOfInterest}" + howManySigmaTitle + thresholds_title1 + thresholds_title2

        self.spectrumTitle = spectrumTitle


    def multiPixel_island(self, bin_width=1,
                          plotSpectrum=False, logarithmic=False,
                          intensity_arb_unit=False,
                          save=True):

        spc_engine = Island_PhotonCounting(indexOfInterest=self.indexOfInterest,
                                           no_photon_adu_thr=self.noP_thresh, sp_adu_thr=self.sp_thresh,
                                           adu_cap=self.adu_cap, adu_offset=self.adu_offset,
                                           removeRows0To_=self.removeTopRows,
                                           howManySigma_thr=self.how_many_sigma, )
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

        moI = spc_engine.imMat

        # The first 3 columns have vertical lines that are due to edge effects
        moI[:, 0:3] = 0


        results_island_dict = spc_engine.operateOnIslands(moI)
        island_list_Cij = results_island_dict["list_countij"]
        energyList_islands = self.lists_energy(island_list_Cij, braggEngine_init=bragg_engine)

        if save:
            index_folder = os.path.join(self.folderpath, str(self.indexOfInterest))
            fileName = f"energy_list_sigma{self.how_many_sigma}.npy"
            array_Elist = np.array(energyList_islands)
            np.save(os.path.join(index_folder, fileName), array_Elist)

        if plotSpectrum:
            self.plotSpectrum(energyList_islands, bin_width, self.spectrumTitle, intensity_arb_unit, logarithmic)

        return energyList_islands, moI

    def shapeSearchEfficient(self):

        Island_PhotonCounting(indexOfInterest=self.indexOfInterest,
                              no_photon_adu_thr=self.noP_thresh, sp_adu_thr=self.sp_thresh,
                              adu_cap=self.adu_cap, adu_offset=self.adu_offset,
                              removeRows0To_=self.removeTopRows,
                              howManySigma_thr=self.how_many_sigma, )
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

    def singlePixel(self,bin_width):
        singlePixel_eng = SinglePixel(self.indexOfInterest,self.how_many_sigma)
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)

        list_countij = singlePixel_eng.photon_count(no_photon_adu_thr=self.noP_thresh, sp_adu_thr=self.sp_thresh,
                                                    adu_offset=self.adu_offset, adu_cap=self.adu_cap)


        e_List = self.lists_energy(list_countij, braggEngine_init=bragg_engine)

        self.plotSpectrum_v2_solidAng(e_List,bin_width,self.spectrumTitle,)

    @staticmethod
    def plotSpectrum(energyList, bin_width, spectrumTitle, intensity_arb_unit=False, logarithmic=False,):

        energyBins = np.arange(1000, 1000+(700 // bin_width + 1) * bin_width, bin_width)

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


    def plotSpectrum_v2_solidAng(self,energyList, bin_width, spectrumTitle,
                        intensity_arb_unit=True, logarithmic=False,):

        index_folder = os.path.join(self.folderpath, str(self.indexOfInterest))
        bin_solid_angle_filepath = os.path.join(index_folder, f"solid_angle_of_binwidth_{bin_width}.npy")

        energyBins = np.arange(1000, 1000+(700 // bin_width + 1) * bin_width, bin_width)

        # create Array of solid A
        if not os.path.exists(bin_solid_angle_filepath):

            arr_solid = solid_angle_array(self.indexOfInterest, bin_width,folderpath=self.folderpath,plotDistribtion=True,
                                          save=True)

        else:
            arr_solid = np.load(bin_solid_angle_filepath)

        photonEnergies = np.array(energyList)
        counts, bins_edges = np.histogram(photonEnergies, energyBins)

        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        sa_norm_count = []
        for solid_ang,count in zip(arr_solid, counts):
            sa_norm_count.append(count/solid_ang)

        arr_norm_counts = np.array(sa_norm_count)

        plt.figure(figsize=(10, 6))
        if intensity_arb_unit:
            countIntensity = []
            for countE, center in zip(arr_norm_counts, bin_centers):
                countIntensity.append(countE * center)
            countIntensity = np.array(countIntensity)
            plt.plot(bin_centers, countIntensity, linestyle='-', color='b')
            plt.ylabel('Intensity (arb. unit)')

        else:
            plt.plot(bin_centers, arr_norm_counts, linestyle='-', color='b')
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

    def islandSpectrum_SolidAngle_with_uncertainty(self,bin_width=1,save=True, diagnosticPrint=False):
        energyList_islands, _ = self.multiPixel_island(bin_width=bin_width, plotSpectrum=False)

        energyBins = np.arange(1000, 1000+(700 // bin_width + 1) * bin_width, bin_width)
        photonEnergies = np.array(energyList_islands)
        count_array, bin_edges = np.histogram(photonEnergies, energyBins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        print(f"count_array: length = {len(count_array)}")
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
        try:
            df_solid_angle["bins"] = df_solid_angle["bins"].apply(ast.literal_eval)
        except ValueError as e:
            print("ValueError when applying string literal to solid angle bins: ", e)

        solidAngle_bins_tuple = df_solid_angle['bins'].values
        solid_angle_vals = df_solid_angle['solid_angle'].values

        solidAngle_bins_center_dict = {}
        for tuple_sa,solid_angle_val in zip(solidAngle_bins_tuple,solid_angle_vals):
            center_val = (tuple_sa[0] + tuple_sa[1])/2
            solidAngle_bins_center_dict[center_val] = solid_angle_val

        solid_angle_normalised_counts = []
        solid_angle_normalised_uncertainty = []
        count_uncertainty = []
        for count, count_center in zip(count_array,bin_centers):
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

            if diagnosticPrint:
                print(f"count = {count}, uncertainty = {uncertainty}, normalised_unc = {normalised_uncertainty}")
                print(f"Normalised count = {normalised_count}, normalised_unc = {normalised_uncertainty}")

        normalised_counts = np.array(solid_angle_normalised_counts)
        normalised_count_unc = np.array(solid_angle_normalised_uncertainty)

        count_unc = np.array(count_uncertainty)

        if save:
            index_spectrum_folder = os.path.join(index_folder, "spectrum")
            if not os.path.exists(index_spectrum_folder):
                os.makedirs(index_spectrum_folder)
            count_filepath = os.path.join(index_spectrum_folder, "count.npy")
            np.save(count_filepath, count_array)

            count_unc_filepath = os.path.join(index_spectrum_folder, "count_unc.npy")
            np.save(count_unc_filepath, count_unc)

            normalised_counts_filepath = os.path.join(index_spectrum_folder, "solidAng_normalised_counts.npy")
            np.save(normalised_counts_filepath, normalised_counts)

            normalised_counts_unc_filepath = os.path.join(index_spectrum_folder, "solidAng_normalised_counts_unc.npy")
            np.save(normalised_counts_unc_filepath, normalised_count_unc)

            bin_center_filepath = os.path.join(index_spectrum_folder, "bin_centers.npy")
            np.save(bin_center_filepath, bin_centers)

        return count_array, count_unc,normalised_counts, normalised_count_unc,bin_centers

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
    def plotSpectrum_count_unc_binCenters(count_array,unc_array,bin_center_array,intensity_arb_unit=True,logarithmic=False,plotUnc=True,
                                          title=""):

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
        plt.title(title)
        plt.show()



# TODO: make a graph with both solid angle corrected and non solid angle corrected!

def accessSavedSpectrums(indexOfInterest,folderpath="stored_variables"):
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    index_spectrum_folder = os.path.join(index_folder, "spectrum")

    count_filepath = os.path.join(index_spectrum_folder, "count.npy")
    count_array = np.load(count_filepath)

    count_unc_filepath = os.path.join(index_spectrum_folder, "count_unc.npy")
    count_unc = np.load(count_unc_filepath)

    normalised_counts_filepath = os.path.join(index_spectrum_folder, "solidAng_normalised_counts.npy")
    normalised_counts = np.load(normalised_counts_filepath)

    normalised_counts_unc_filepath = os.path.join(index_spectrum_folder, "solidAng_normalised_counts_unc.npy")
    normalised_counts_unc = np.load(normalised_counts_unc_filepath)

    bin_center_filepath = os.path.join(index_spectrum_folder, "bin_centers.npy")
    bin_centers = np.load(bin_center_filepath)

    return count_array,count_unc,normalised_counts,normalised_counts_unc,bin_centers



def dict_ij_perEnergyBin(index_of_interest, bin_width=1, folderpath="stored_variables"):
    print("-" * 30)
    print("Creating dictionary of ij coordinates in each bin")

    index_folder = os.path.join(folderpath, str(index_of_interest))
    energy_filepath = os.path.join(index_folder, "energy_of_pixel.npy")
    energy_of_pixelMat = np.load(energy_filepath)

    energyBands = np.arange(1000, 1000+(700 // bin_width + 1) * bin_width, bin_width)

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

def solid_angle_array(index_of_interest, bin_width=1, folderpath="stored_variables",
                               plotDistribtion=False, save=True):
    print("-" * 30)
    print("Creating dictionary of total Solid Angle captured for each bin")
    print("Start:", time.strftime("%H:%M:%S", time.localtime()))
    dict_ij_perBin = dict_ij_perEnergyBin(index_of_interest, bin_width=bin_width, folderpath=folderpath)

    index_folder = os.path.join(folderpath, str(index_of_interest))
    solidAngle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
    solidAng_mat = np.load(solidAngle_filepath)

    energy_bins = np.arange(1000, 1000+(700 // bin_width + 1) * bin_width, bin_width)
    # print(energy_bins)
    arr_solidAng = np.zeros(len(energy_bins)-1)

    for (lower_bound, upper_bound), list_ij in dict_ij_perBin.items():
        totalSolidAngle = 0

        for i_idx, j_idx in list_ij:
            totalSolidAngle += solidAng_mat[i_idx, j_idx]

        # Find the index corresponding to lower_bound
        # bin_idx = np.where(energy_bins == lower_bound)[0][0]
        bin_idx = np.where(energy_bins == lower_bound)[0]

        # Store totalSolidAngle in the correct bin index
        arr_solidAng[bin_idx] = totalSolidAngle

    print("Finish:", time.strftime("%H:%M:%S", time.localtime()))

    if save:
        bin_solid_angle_filepath = os.path.join(index_folder, f"solid_angle_of_binwidth_{bin_width}.npy")
        np.save(bin_solid_angle_filepath, arr_solidAng)

    if plotDistribtion:
        bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, arr_solidAng, marker='o', linestyle='-', color='b', label="Total Solid Angle")
        plt.xlabel("Energy Bin Center (eV)")
        plt.ylabel("Total Solid Angle")
        plt.title("Distribution of Solid Angle")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    return arr_solidAng



def collect_savedSpectrums(bin_width=1, list_indices=list_data, folderpath="stored_variables",save=True):
    topBin_dif = 700
    topBin_dif_binWidth_rounded = 700 // bin_width

    energyBins = np.arange(1000, (topBin_dif_binWidth_rounded + 1) * bin_width, bin_width)
    bin_centers = (energyBins[:-1] + energyBins[1:]) / 2

    dict_counts = {}

    for center in bin_centers:
        dict_counts[center] = {"count": 0,
                               "count_unc": 0,
                               "normalised_counts": 0,
                               "normalised_counts_unc": 0,}

    for indexOI in list_indices:
        try:
            count_array,count_unc,normalised_counts,normalised_counts_unc,bin_centers = accessSavedSpectrums(indexOI,folderpath=folderpath)

            for center_,count_,count_unc_,norm_count,norm_count_unc in zip(bin_centers, count_array, count_unc,normalised_counts,normalised_counts_unc):
                dict_center_ = dict_counts[center_]
                dict_center_["count"] += count_
                dict_center_["count_unc"] += count_unc_
                dict_center_["normalised_counts"] += norm_count
                dict_center_["normalised_counts_unc"] += norm_count_unc


        except FileNotFoundError as e:
            print(f"Index = {indexOI}, FileNotFoundError: {e}")


        def create_arrays():
            count_list = []
            count_unc_list = []
            count_normalised_list = []
            count_normalised_unc_list = []
            bin_centers_list = []

            for key in dict_counts.keys():
                bin_centers_list.append(key)
                dict_center_ = dict_counts[key]

                count_list.append(dict_center_["count"])
                count_unc_list.append(dict_center_["count_unc"])
                count_normalised_list.append(dict_center_["normalised_counts"])
                count_normalised_unc_list.append(dict_center_["normalised_counts_unc"])

            array_bin_centers_ = np.array(bin_centers_list)
            array_count_ = np.array(count_list)
            array_count_unc_ = np.array(count_unc_list)
            array_norm_count_ = np.array(count_normalised_list)
            array_norm_count_unc_ = np.array(count_normalised_unc_list)

            return array_bin_centers_, array_count_, array_count_unc_, array_norm_count_, array_norm_count_unc_


        if save:
            tot_spec_folder = os.path.join(folderpath,f"Total_Spectrum_BinWidth_{bin_width}")
            if not os.path.exists(tot_spec_folder):
                os.makedirs(tot_spec_folder)

            count_fp = os.path.join(tot_spec_folder, "counts.npy")
            count_unc_fp = os.path.join(tot_spec_folder, "counts_unc.npy")
            norm_count_fp = os.path.join(tot_spec_folder, "normalised_counts.npy")
            norm_count_unc_fp = os.path.join(tot_spec_folder, "normalised_counts_unc.npy")
            bin_center_fp = os.path.join(tot_spec_folder, "bin_centers.npy")

            array_bin_centers, array_count, array_count_unc, array_norm_count, array_norm_count_unc = create_arrays()
            np.save(bin_center_fp,array_bin_centers)

            np.save(count_fp,array_count)
            np.save(count_unc_fp,array_count_unc)

            np.save(norm_count_fp,array_norm_count)
            np.save(norm_count_unc_fp,array_norm_count_unc)

def access_totalSavedSpectrums(bin_width=1,folderpath="stored_variables"):
    tot_spec_folder = os.path.join(folderpath, f"Total_Spectrum_BinWidth_{bin_width}")

    count_fp = os.path.join(tot_spec_folder, "counts.npy")
    count_unc_fp = os.path.join(tot_spec_folder, "counts_unc.npy")
    norm_count_fp = os.path.join(tot_spec_folder, "normalised_counts.npy")
    norm_count_unc_fp = os.path.join(tot_spec_folder, "normalised_counts_unc.npy")
    bin_center_fp = os.path.join(tot_spec_folder, "bin_centers.npy")

    tot_bin_centers = np.load(bin_center_fp)

    tot_count = np.load(count_fp)
    tot_count_unc = np.load(count_unc_fp)

    tot_norm_count = np.load(norm_count_fp)
    tot_norm_count_unc = np.load(norm_count_unc_fp)

    return tot_bin_centers,tot_count,tot_count_unc,tot_norm_count,tot_norm_count_unc

def plot_individual_Saved_spec(indexOI,folderpath="stored_variables"):
    count,count_unc,normalised_counts,normalised_counts_unc,bin_centers = accessSavedSpectrums(indexOI,folderpath)

    # for item_array in [count,count_unc,normalised_counts,normalised_counts_unc,bin_centers]:
    #     print(len(item_array))

    # print(normalised_counts)
    # print("\n\nUncertainty: \n\n")
    # print(normalised_counts_unc)


    spectrum = Spectrum_old(indexOI, removeTopRows=0,
                            how_many_sigma=2, no_photon_adu_thr=80, sp_adu_thr=150, adu_offset=40, adu_cap=1650,
                            )

    spectrum.plotSpectrum_count_unc_binCenters(count_array=count,unc_array=count_unc,bin_center_array=bin_centers,
                                               title=f"Image {indexOI} spectrum without Solid Angle Corrections",
                                               plotUnc=False,intensity_arb_unit=False)

    # spectrum.plotSpectrum_count_unc_binCenters(count_array=normalised_counts, unc_array=normalised_counts_unc, bin_center_array=bin_centers,
    #                                            title=f"Image {indexOI} spectrum with Solid Angle Corrections",
    #                                            plotUnc=False,intensity_arb_unit=True)


def check_spec(indexOI,folderpath="stored_variables",testPrint=False,save=False,remove_top_rows=0):
    spectrum = Spectrum_old(indexOI, removeTopRows=remove_top_rows,
                            how_many_sigma=4, no_photon_adu_thr=100, sp_adu_thr=150, adu_offset=40, adu_cap=1650,
                            )


    (count, count_unc,
     normalised_counts, normalised_count_unc,
     bin_centers) = spectrum.islandSpectrum_SolidAngle_with_uncertainty(bin_width=1,save=save,diagnosticPrint=False)

    if testPrint:

        howManySigmaTitle_ = f'\nConsidering values {spectrum.how_many_sigma} above the mean'
        thresholds_title1_ = f'\nAcceptance Region: {spectrum.noP_thresh} < ADU total < {spectrum.adu_cap}'
        thresholds_title2_ = f'\nSingle Photon ADU = {spectrum.sp_thresh} with allowed offset = {spectrum.adu_offset}'
        if remove_top_rows != 0:
            removeTR_title = f'\n{remove_top_rows} rows removed from the top'

            spectrumTitle_ = f"Photon Energy Spectrum for image {spectrum.indexOfInterest}" + howManySigmaTitle_ + thresholds_title1_ + thresholds_title2_ + removeTR_title
        else:
            spectrumTitle_ = f"Photon Energy Spectrum for image {spectrum.indexOfInterest}" + howManySigmaTitle_ + thresholds_title1_ + thresholds_title2_

        spectrum.plotSpectrum_count_unc_binCenters(count_array=normalised_counts, unc_array=normalised_count_unc,
                                                   bin_center_array=bin_centers, plotUnc=False,
                                                   title=spectrumTitle_)


def plot_total_spectrum(bin_width=1,folderpath="stored_variables"):
    tot_bin_centers, tot_count, tot_count_unc, tot_norm_count, tot_norm_count_unc = access_totalSavedSpectrums(bin_width=bin_width,folderpath=folderpath)

    spectrum = Spectrum_old(8, removeTopRows=0,
                            how_many_sigma=2, no_photon_adu_thr=80, sp_adu_thr=150, adu_offset=40, adu_cap=1650,
                            )

    spectrum.plotSpectrum_count_unc_binCenters(count_array=tot_count, unc_array=tot_count_unc, bin_center_array=tot_bin_centers,)

    spectrum.plotSpectrum_count_unc_binCenters(count_array=tot_norm_count, unc_array=tot_norm_count_unc,bin_center_array=tot_bin_centers,
                                               )


# TODO: get comparison of count and normalised + intensity count

# TODO: Try 1.5 bin width, make code more modular for that


if __name__ == "__main__":
    # collect_savedSpectrums()

    def generate_all_individual_spectrums():

        for index_ in list_data:
            spectrum = Spectrum_old(index_, removeTopRows=0,
                                    how_many_sigma=2, no_photon_adu_thr=80, sp_adu_thr=150, adu_offset=40, adu_cap=1600,
                                    )
            spectrum.islandSpectrum_SolidAngle_with_uncertainty(bin_width=1, save=True, diagnosticPrint=False)


    def plot_all_individual_spectrums():
        for index_OI in list_data:
            plot_individual_Saved_spec(index_OI)


    def singlePixel_spec(indexOI):
        print("singlePixel_spec")
        spectrum = Spectrum_old(indexOI, removeTopRows=0,
                                how_many_sigma=2, no_photon_adu_thr=80, sp_adu_thr=150, adu_offset=40, adu_cap=1600,
                                )
        spectrum.singlePixel(bin_width=1)


    # singlePixel_spec(11)


    def setshapes_energy_lists(indexOI):
        spec_eng = Spectrum(indexOI)
        isl_filename = "energy_list_island_2photonMax.npy"
        spec_eng.energy_list_island(save=True,
                                    save_filename=isl_filename)
        spec_eng.spectrum_set_then_island(bin_width=1,island_filename=isl_filename)


    setshapes_energy_lists(11)

    # plot_all_individual_spectrums()

    # generate_all_individual_spectrums()

    # collect_savedSpectrums()
    # plot_total_spectrum()

    # check_spec(11,testPrint=True,remove_top_rows=0)

    # printSaved_spec(11,)



    def solidAngle(indexOI=11):
        spectrum_eng = Spectrum_old(indexOfInterest=indexOI, folderpath="stored_variables", how_many_sigma=2, )
        # spectrum_eng.islandSpectrum_SolidAngle_with_uncertainty(bin_width=1, save=True)
        spectrum_eng.plotSpectrum_solid_angle(bin_width=1,intensity_arb_unit=True,)

    # solidAngle(8)


    # solid_angle_per_energy_bin(8,1,plotDistribtion=True,save=False)



    pass
