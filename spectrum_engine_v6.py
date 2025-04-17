from calibrate_geometry_v4 import *
from spc_engine_5 import *
import time
import pandas as pd
import ast

def spectrum_folderpath(folderpath_, index_of_interest_):
    index_folder = os.path.join(folderpath_, str(index_of_interest_))
    spc_folderpath = os.path.join(index_folder, "spectrum")
    if not os.path.exists(spc_folderpath):
        os.makedirs(spc_folderpath)
    return spc_folderpath

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

    # Initialising folder locals:
    index_folder = os.path.join(folderpath, str(index_of_interest))
    solidAngle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
    solidAng_mat = np.load(solidAngle_filepath)

    bin_solid_angle_filename = f"solid_angle_of_binwidth_{bin_width}.xlsx"
    bin_solid_angle_filepath = os.path.join(spectrum_folderpath(folderpath_=folderpath, index_of_interest_=index_of_interest), bin_solid_angle_filename)

    try:
        solid_angle_df = pd.read_excel(bin_solid_angle_filepath,)
        # Convert bins back to tuples
        solid_angle_df['bins'] = solid_angle_df['bins'].apply(ast.literal_eval)
        totalSolidAngle_dict = dict(solid_angle_df.values)

    except FileNotFoundError as e:
        print("Saved File not found, creating and saving:")
        dict_ij_perBin = dict_ij_perEnergyBin(index_of_interest, bin_width=bin_width, folderpath=folderpath)

        print("-" * 30)
        print("Creating dictionary of total Solid Angle captured for each bin")
        print("Start:", time.strftime("%H:%M:%S", time.localtime()))


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
        if save:
            bin_solid_angle_filename = f"solid_angle_of_binwidth_{bin_width}.xlsx"
            bin_solid_angle_filepath = os.path.join(spectrum_folderpath(folderpath_=folderpath,index_of_interest_=index_of_interest), bin_solid_angle_filename)
            solid_angle_df.to_excel(bin_solid_angle_filepath, index=False)

    if plotDistribtion:
        bin_bounds = np.array(list(totalSolidAngle_dict.keys()))
        bin_centers = (bin_bounds[:, 0] + bin_bounds[:, 1]) / 2
        total_solid_angles = np.array(list(totalSolidAngle_dict.values()))

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, total_solid_angles,
                 # marker='.', markersize=1,
                 linestyle='-', color='b', label="Total Solid Angle")
        plt.xlabel("Energy Bin Center (eV)")
        plt.ylabel("Total Solid Angle (str)")
        plt.title(f"Image {index_of_interest}: Distribution of Solid Angle with bin width = {bin_width} eV")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    return totalSolidAngle_dict

class Spectrum_Island:
    def __init__(self, indexOfInterest,
                 geo_engine=None,
                 spc_engine=None,
                 bin_width=1,
                 folderpath="stored_variables",
                 spectrumTitle=None,
                 use_saved_values=True,
                 diagnostics=False,
                 declareVars=True):

        print("-" * 30)
        print("Spectrum Class Initialised with:")
        print("indexOfInterest = ", indexOfInterest)

        if geo_engine is None:
            geo_engine = geo_engine_withSavedParams(indexOfInterest,declareVars)


        if spc_engine is None:
            self.spc_engine = Island_PhotonCounting(indexOfInterest, no_photon_adu_thr=100, sp_adu_thr=150, adu_offset=30, adu_cap=5000,
                 removeRows0To_=0, howManySigma_thr=2,declareVars=declareVars )
        else:
            self.spc_engine = spc_engine



        crystal_pitch = geo_engine.crystal_pitch
        crystal_roll = geo_engine.crystal_roll
        camera_pitch = geo_engine.camera_pitch
        camera_roll = geo_engine.camera_roll
        r_camera_spherical = geo_engine.r_camera_spherical

        self.indexOfInterest = indexOfInterest

        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical

        self.bragg_eng = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)
        self.folderpath = folderpath

        self.bin_width = bin_width

        if spectrumTitle is None:
            howManySigmaTitle = f'\nConsidering values {self.spc_engine.howManySigma} above the mean'
            # thresholds_title1 = f'\nAcceptance Region: {self.noP_thresh} < ADU total < {self.adu_cap}'
            # thresholds_title2 = f'\nSingle Photon ADU = {self.sp_thresh} with allowed offset = {self.adu_offset}'
            spectrumTitle = f"Photon Energy Spectrum for image {self.indexOfInterest}" + howManySigmaTitle
                             # + thresholds_title1 + thresholds_title2)

        self.spectrumTitle = spectrumTitle
        self.use_saved_values = use_saved_values
        self.diagnostics = diagnostics

        self.pc_eng_error_fraction = Unit_testing(self.indexOfInterest).find_uncertainty_in_spc_engine(self.diagnostics)

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
    def normalise_array(array):
        array_max = array.max()
        norm_array = array / array_max
        return norm_array

    @staticmethod
    def normalise_array_and_uncertainty(arr_counts,arr_uncertainty):
        max_val = np.max(arr_counts)
        arr_norm_counts = arr_counts / max_val
        arr_norm_unc = np.array(arr_uncertainty) / max_val

        return arr_norm_counts, arr_norm_unc

    def bin_edges_array(self):
        return np.arange(1000, 1000 + (700 // self.bin_width + 1) * self.bin_width, self.bin_width)

    @staticmethod
    def bin_centers_array(bins_edges):
        return (bins_edges[:-1] + bins_edges[1:]) / 2

    def solid_angle_array(self, diagnostic=False):
        energyBins = self.bin_edges_array()
        bin_centers = self.bin_centers_array(energyBins)

        solid_angle_dict = solid_angle_per_energy_bin(self.indexOfInterest,bin_width=self.bin_width,folderpath=self.folderpath)

        bin_tuples = np.array(list(solid_angle_dict.keys()))
        l_sa_bin_centers = (bin_tuples[:, 0] + bin_tuples[:, 1]) / 2
        solid_angle_vals = np.array(list(solid_angle_dict.values()))

        if diagnostic:
            print("length l_sa_bin_centers", len(l_sa_bin_centers))
            print("length solid_angle_vals", len(solid_angle_vals))


        if len(l_sa_bin_centers) == len(bin_centers):
            l_solid_angle = solid_angle_vals
        else:
            l_solid_angle = np.zeros_like(bin_centers, dtype=solid_angle_vals.dtype)
            mask = np.isin(bin_centers, l_sa_bin_centers)

            l_solid_angle[mask] = solid_angle_vals[np.searchsorted(l_sa_bin_centers, bin_centers[mask])]

        if diagnostic:
            print("length bin center", len(bin_centers))
            print("length l_solid_angle", len(l_solid_angle))
            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(bin_centers, l_solid_angle, marker='.', linestyle='-', color='b', label="Total Solid Angle",
                     markersize=3)
            plt.xlabel("Energy Bin Center (eV)")
            plt.ylabel("Total Solid Angle")
            plt.title(f"Distribution of Solid Angle with bin width = {self.bin_width} eV")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.show()


        return np.array(l_solid_angle)

    def energy_list_island(self,save=True):

        filename = f"energy_list.npy"
        filepath = os.path.join(spectrum_folderpath(self.folderpath,self.indexOfInterest), filename)

        if self.use_saved_values:
            try:
                arr_energy_list = np.load(filepath)
            except FileNotFoundError:
                print("File not yet saved. Creating array and saving in index/spectrum folder")


                output_dict = self.spc_engine.operateOnIslands()
                count_ij_island = output_dict["list_countij"]

                energy_list_island = self.lists_energy(list_countij=count_ij_island,braggEngine_init=self.bragg_eng)
                arr_energy_list = np.array(energy_list_island)

                if save:
                    np.save(filepath, arr_energy_list)

        else:
            output_dict = self.spc_engine.operateOnIslands()
            count_ij_island = output_dict["list_countij"]

            energy_list_island = self.lists_energy(list_countij=count_ij_island, braggEngine_init=self.bragg_eng)
            arr_energy_list = np.array(energy_list_island)

        return arr_energy_list

    def compute_uncertainty(self, hist_count):
        unc_spc_eng = hist_count * self.pc_eng_error_fraction
        unc_poisson = np.sqrt(hist_count)

        # print("PC engine uncertainty", unc_spc_eng)
        # print("Poisson Uncertainty: ", unc_poisson)
        # adding in quadrature
        return np.sqrt(unc_spc_eng**2 + unc_poisson**2)

    def non_corrected_array(self,normalised=True):
        energyList = self.energy_list_island()
        energyBins = self.bin_edges_array()
        photonEnergies = np.array(energyList)
        hist_counts, bins_edges = np.histogram(photonEnergies, energyBins)
        bin_centers = self.bin_centers_array(bins_edges)

        l_uncertainty = []
        for count_hist in hist_counts:
            l_uncertainty.append(self.compute_uncertainty(count_hist))

        if normalised:
            arr_return_hist_counts, arr_return_unc = self.normalise_array_and_uncertainty(
                hist_counts, l_uncertainty)
        else:
            arr_return_hist_counts = hist_counts
            arr_return_unc = np.array(l_uncertainty)

        return bin_centers, arr_return_hist_counts, arr_return_unc

    def corrected_count_array_unity(self, ):
        energyList = self.energy_list_island()
        energyBins = self.bin_edges_array()
        photonEnergies = np.array(energyList)
        hist_counts, bins_edges = np.histogram(photonEnergies, energyBins)
        bin_centers = self.bin_centers_array(bins_edges)

        arr_solid_angle = self.solid_angle_array()

        l_corrected_count = []
        l_corrected_unc = []

        for energy_bin_center, hist_count, solid_angle in zip(bin_centers,hist_counts, arr_solid_angle):
            if hist_count == 0:
                l_corrected_count.append(0)
                l_corrected_unc.append(0)
                continue

            unc = self.compute_uncertainty(hist_count)

            intensity_corrected_count = hist_count * energy_bin_center
            intensity_corrected_unc = unc * energy_bin_center

            solid_angle_corrected_count = intensity_corrected_count / solid_angle
            solid_angle_corrected_unc = intensity_corrected_unc / solid_angle

            l_corrected_count.append(solid_angle_corrected_count)
            l_corrected_unc.append(solid_angle_corrected_unc)

        arr_corrected_count = np.array(l_corrected_count)
        arr_corrected_unc = np.array(l_corrected_unc)

        norm_corrected_count, norm_corrected_unc = self.normalise_array_and_uncertainty(arr_corrected_count,arr_corrected_unc)

        return bin_centers, norm_corrected_count, norm_corrected_unc

    def corrected_count_array(self, ):
        energyList = self.energy_list_island()
        energyBins = self.bin_edges_array()
        photonEnergies = np.array(energyList)
        hist_counts, bins_edges = np.histogram(photonEnergies, energyBins)
        bin_centers = self.bin_centers_array(bins_edges)

        arr_solid_angle = self.solid_angle_array()

        l_corrected_count = []
        l_corrected_unc = []

        for energy_bin_center, hist_count, solid_angle in zip(bin_centers,hist_counts, arr_solid_angle):
            if hist_count == 0:
                l_corrected_count.append(0)
                l_corrected_unc.append(0)
                continue

            unc = self.compute_uncertainty(hist_count)

            intensity_corrected_count = hist_count * energy_bin_center
            intensity_corrected_unc = unc * energy_bin_center

            solid_angle_corrected_count = intensity_corrected_count / solid_angle
            solid_angle_corrected_unc = intensity_corrected_unc / solid_angle

            l_corrected_count.append(solid_angle_corrected_count)
            l_corrected_unc.append(solid_angle_corrected_unc)

        arr_corrected_count = np.array(l_corrected_count)
        arr_corrected_unc = np.array(l_corrected_unc)

        return bin_centers, arr_corrected_count, arr_corrected_unc


    def spectrum_energy_list_island(self,intensity_arb_unit=False, logarithmic=False,):

        energyList = self.energy_list_island()

        energyBins = self.bin_edges_array()

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
        plt.title(self.spectrumTitle)
        if logarithmic:
            plt.yscale('log')
        plt.grid(True)
        plt.show()

    def spectrum_normalised_corrected_count(self,logarithmic=False,xBounds=(1100,1600)):
        bin_centers, norm_corrected_count, norm_corrected_unc = self.corrected_count_array_unity()

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, norm_corrected_count, linestyle='-', color='b', label='Corrected Count')

        # Add shaded uncertainty region
        plt.fill_between(
            bin_centers,
            norm_corrected_count - norm_corrected_unc,
            norm_corrected_count + norm_corrected_unc,
            color='b',
            alpha=0.2,
            label='Uncertainty'
        )

        plt.ylabel('Intensity (arb. unit)')
        plt.xlabel('Energy (eV)')
        plt.title(self.corrected_count_spectrum_title())
        if logarithmic:
            plt.yscale('log')
        plt.xlim(xBounds[0], xBounds[1])
        plt.grid(True)
        plt.legend()
        plt.show()

    def spectrum_compare_normalised_corrected_not_corrected(self,logarithmic=False,xBounds = (1100,1600)):

        bin_centers, norm_corrected_count, norm_corrected_unc = self.corrected_count_array_unity()
        bin_centers, arr_return_hist_counts, arr_return_unc = self.non_corrected_array()

        plt.figure(figsize=(10, 6))

        # Corrected Count
        plt.plot(bin_centers, norm_corrected_count, linestyle='-', color='b', label='Corrected Intensity')
        plt.fill_between(bin_centers,
            norm_corrected_count - norm_corrected_unc,
            norm_corrected_count + norm_corrected_unc,
            color='b',alpha=0.2,label='Corrected Intensity Uncertainty'
        )
        plt.plot(bin_centers, arr_return_hist_counts, linestyle='-', color='g', label='PC Engine Count')
        plt.fill_between(bin_centers,
                         arr_return_hist_counts - arr_return_unc,
                         arr_return_hist_counts + arr_return_unc,
                         color='g', alpha=0.2, label='PC Engine Count Uncertainty'
                         )
        plt.xlabel('Energy (eV)')
        plt.title(f"Photon Count and Corrected Spectrum for Image {self.indexOfInterest}")
        if logarithmic:
            plt.yscale('log')
        plt.xlim(xBounds[0], xBounds[1])
        plt.ylim(ymax=1.2)
        plt.grid(True)
        plt.ylabel('Intensity (arb. unit)')
        plt.legend()
        plt.show()

    def corrected_count_spectrum_title(self):
        howManySigmaTitle = f'\nConsidering values {self.spc_engine.howManySigma} above the mean'
        main_title = f"Solid Angle Normalised Photon Intensity Spectrum for Image {self.indexOfInterest}"
        return main_title + howManySigmaTitle


# Only really used for graphics

class Spectrum_with_Preset_shapes:
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


class Combine_Spectra:
    def __init__(self,bin_width, list_of_indices,folderpath="stored_variables", diagnostics=False):
        self.list_of_indices = list_of_indices
        self.folderpath = folderpath
        self.bin_width = bin_width
        self.diagnostics = diagnostics

    def total_spectrum_folderpath(self):
        avg_spectra_folder = os.path.join(self.folderpath, "Averaged_spectra")
        if not os.path.exists(avg_spectra_folder):
            os.makedirs(avg_spectra_folder)
        return avg_spectra_folder

    def bin_width_sub_folderpath(self):
        bin_w_folderpath = os.path.join(self.total_spectrum_folderpath(), f"bin_width_{self.bin_width}")
        if not os.path.exists(bin_w_folderpath):
            os.makedirs(bin_w_folderpath)
        return bin_w_folderpath

    def corrected_count_and_unc_filepaths(self):
        bw_folderpath = self.bin_width_sub_folderpath()
        count_filename = f"combined_corrected_counts.npy"
        unc_filename = "combined_corrected_uncertainties.npy"
        return os.path.join(bw_folderpath, count_filename), os.path.join(bw_folderpath, unc_filename)


    def plot_each_spectra(self,bin_width):
        for index_OI in self.list_of_indices:
            specIsland_eng = Spectrum_Island(index_OI,folderpath=self.folderpath,bin_width=bin_width)
            specIsland_eng.spectrum_compare_normalised_corrected_not_corrected()

    def average_normalised_corrected_spectra(self, plot=False,use_temp_fileLocal=False):

        if use_temp_fileLocal:
            bw_folderpath = self.bin_width_sub_folderpath()
            count_filename = f"reduced_list_combined_corrected_counts.npy"
            unc_filename = "reduced_list_combined_corrected_uncertainties.npy"
            fp_count=os.path.join(bw_folderpath, count_filename)
            fp_unc=os.path.join(bw_folderpath, unc_filename)
        else:
            fp_count, fp_unc = self.corrected_count_and_unc_filepaths()


        bin_edges_ = Spectrum_Island(8, bin_width=self.bin_width, declareVars=False).bin_edges_array()
        bin_centers_ = (bin_edges_[:-1] + bin_edges_[1:]) / 2

        try:
            arr_mean_norm_count = np.load(fp_count)
            arr_mean_norm_unc = np.load(fp_unc)
        except FileNotFoundError as e:
            print("Files not yet saved: {}".format(e))


            combined_counts = []
            combined_uncs = []


            # create a matrix where the column
            for index_OI in self.list_of_indices:
                specIsland_eng = Spectrum_Island(index_OI,folderpath=self.folderpath,bin_width=self.bin_width,
                                                 declareVars=False)
                bin_centers, norm_corrected_count, norm_corrected_unc = specIsland_eng.corrected_count_array_unity()

                if self.diagnostics:
                    print("count length: ", len(norm_corrected_count))
                    print("unc length:", len(norm_corrected_unc))

                combined_counts.append(norm_corrected_count)
                combined_uncs.append(norm_corrected_unc)

            # Convert to matrices
            mat_combined_counts = np.array(combined_counts)
            mat_combined_uncs = np.array(combined_uncs)

            if self.diagnostics:
                print("mat_combined_counts shape: ", mat_combined_counts.shape)
                print("mat_combined_uncs shape: ", mat_combined_uncs.shape)

            arr_mean_norm_count = np.mean(mat_combined_counts,axis=0)
            arr_mean_norm_unc = np.sqrt(np.sum(mat_combined_uncs,axis=0)) / mat_combined_uncs.shape[0]

            np.save(fp_count,arr_mean_norm_count)
            np.save(fp_unc,arr_mean_norm_unc)



        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(bin_centers_, arr_mean_norm_count, linestyle='-', color='b', label='Corrected Count')

            # Add shaded uncertainty region
            plt.fill_between(
                bin_centers_,
                arr_mean_norm_count - arr_mean_norm_unc,
                arr_mean_norm_count + arr_mean_norm_unc,
                color='b',
                alpha=0.2,
                label='Uncertainty'
            )

            plt.ylabel('Intensity (arb. unit)')
            plt.xlabel('Energy (eV)')
            plt.title(f"Averaged Corrected Spectra with bin width {self.bin_width}")
            # plt.xlim(xBounds[0], xBounds[1])
            plt.grid(True)
            plt.legend()
            plt.show()

        return bin_centers_, arr_mean_norm_count, arr_mean_norm_unc

    def plot_average_normalised_corrected_spectra(self):
        bin_centers_, arr_mean_norm_count, arr_mean_norm_unc = self.average_normalised_corrected_spectra()
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers_, arr_mean_norm_count, linestyle='-', color='b', label='Corrected Count')

        # Add shaded uncertainty region
        plt.fill_between(
            bin_centers_,
            arr_mean_norm_count - arr_mean_norm_unc,
            arr_mean_norm_count + arr_mean_norm_unc,
            color='b',
            alpha=0.2,
            label='Uncertainty'
        )

        plt.ylabel('Intensity (arb. unit)')
        plt.xlabel('Energy (eV)')
        plt.title(f"Averaged Corrected Spectra with bin width {self.bin_width}")
        # plt.xlim(xBounds[0], xBounds[1])
        plt.grid(True)
        plt.legend()
        plt.show()

    def compare_all_spectra_not_normalised_to_unity(self,logarithmic=False):
        plt.figure(figsize=(10, 6))
        for index_OI in self.list_of_indices:
            spec_eng = Spectrum_Island(index_OI,folderpath=self.folderpath,bin_width=self.bin_width,declareVars=False)
            bin_centers, norm_corrected_count, norm_corrected_unc = spec_eng.corrected_count_array()
            plt.plot(bin_centers, norm_corrected_count, linestyle='-', label= f'Image {index_OI}')
            # Add shaded uncertainty region
            plt.fill_between(
                bin_centers,
                norm_corrected_count - norm_corrected_unc,
                norm_corrected_count + norm_corrected_unc,
                alpha=0.2,
                label='Uncertainty'
            )

        plt.ylabel('Intensity (arb. unit)')
        plt.xlabel('Energy (eV)')
        plt.title(f"Solid Angle Normalised Energy Spectra")
        plt.grid(True)
        if logarithmic:
            plt.yscale('log')

        plt.legend()
        plt.show()


    def compare_im8_11(self,logarithmic=False):
        spec8 = Spectrum_Island(8, bin_width=self.bin_width)
        spec11 = Spectrum_Island(11, bin_width=self.bin_width)
        bin_centers, norm_corrected_count8, norm_corrected_unc8 = spec8.corrected_count_array()
        bin_centers, norm_corrected_count11, norm_corrected_unc11 = spec11.corrected_count_array()

        plt.figure(figsize=(10, 6))

        # im 8
        plt.plot(bin_centers, norm_corrected_count8, linestyle='-', color='b', label='Image 8')
        # Add shaded uncertainty region
        plt.fill_between(
            bin_centers,
            norm_corrected_count8 - norm_corrected_unc8,
            norm_corrected_count8 + norm_corrected_unc8,
            color='b',
            alpha=0.2,
            label='Uncertainty'
        )
        # im 11
        plt.plot(bin_centers, norm_corrected_count11, linestyle='-', color='g', label='Image 11')
        # Add shaded uncertainty region
        plt.fill_between(
            bin_centers,
            norm_corrected_count11 - norm_corrected_unc11,
            norm_corrected_count11 + norm_corrected_unc11,
            color='g',
            alpha=0.2,
            label='Uncertainty'
        )

        plt.ylabel('Intensity (arb. unit)')
        plt.xlabel('Energy (eV)')
        plt.title(f"Solid Angle Normalised Energy Spectra of Images 8 and 11")
        plt.grid(True)
        plt.legend()
        if logarithmic:
            plt.yscale('log')
        plt.show()



if __name__ == "__main__":

    Combine_Spectra(bin_width=1.5,list_of_indices=list_good_data).compare_im8_11()

    def investigate_unc(list_indices=list_good_data):
        for im_idx in list_indices:
            spec_eng = Spectrum_Island(im_idx)
            spec_eng.corrected_count_array_unity()

    # investigate_unc()

    # Spectrum_Island(8,bin_width=1.5).spectrum_compare_normalised_corrected_not_corrected()
    # Spectrum_Island(11, bin_width=1.5).spectrum_compare_normalised_corrected_not_corrected()

    # Combine_Spectra(1,list_data,diagnostics=True).average_normalised_corrected_spectra(True)

    # Spectrum_Island(1,bin_width=1.5).solid_angle_array(True)

    # Combine_Spectra(1.5, [1,2,4,7,8,14],diagnostics=True).average_normalised_corrected_spectra(plot=True,use_temp_fileLocal=True)

    def investigate_new_spectrum_model(indexOI):
        spc_eng = Island_PhotonCounting(indexOfInterest=indexOI, no_photon_adu_thr=100, sp_adu_thr=150, adu_offset=30, adu_cap=5000,
                 removeRows0To_=0, howManySigma_thr=2,how_many_more_sigma=2,diagnostic_print=False)
        spectrum_eng = Spectrum_Island(indexOfInterest=indexOI,geo_engine=None,spc_engine=spc_eng,
                        bin_width=1,use_saved_values=False)
        spectrum_eng.spectrum_compare_normalised_corrected_not_corrected()


    # investigate_new_spectrum_model(11)


    # solid_angle_per_energy_bin(8,1,plotDistribtion=True,)
    # solid_angle_per_energy_bin(1, 1.5, plotDistribtion=True, )
    # collect_savedSpectrums()

    # check_spec(11,testPrint=True)

    # plot_individual_Saved_spec(11)



    # singlePixel_spec(11)


    def setshapes_energy_lists(indexOI):
        spec_eng = Spectrum_with_Preset_shapes(indexOI)
        isl_filename = "energy_list_island_2photonMax.npy"
        spec_eng.energy_list_island(save=True,
                                    save_filename=isl_filename)
        spec_eng.spectrum_set_then_island(bin_width=1,island_filename=isl_filename)


    # setshapes_energy_lists(11)





    pass
