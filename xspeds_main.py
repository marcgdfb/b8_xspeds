from spectrum_engine_v6 import *


def main(list_image_indices=list_good_data, saved_folderpath="stored_variables", bin_width=1.5):
    """
    :param list_image_indices: list of image indices with relevant data
    :param saved_folderpath: Folderpath for where values are/will be saved
    :param bin_width: Bin width (eV) for the spectrum. Suggested value for the data used to construct
    this module is 1.5 eV
    :return:
    """
    print("main:")
    print("list_image_indices:", list_image_indices)

    if not os.path.exists(saved_folderpath):
        print("Creating folder: ", saved_folderpath)
        os.makedirs(saved_folderpath)

    for index_image in list_image_indices:
        print(f"Working on image {index_image}\n")
        print_current_datetime()
        start_time = time.time()

        # Checking for saved geometry calibration data. Saving in the case that it doesn't exist
        try:
            access_saved_ellipse(index_image,saved_folderpath)
        except FileNotFoundError:
            fit_ellipse(index_image, plot_Results=False, folderpath=saved_folderpath)
        try:
            access_saved_geometric(index_image,saved_folderpath)
        except FileNotFoundError:
            fit_geometry_to_ellipse(index_image,folderpath=saved_folderpath)

        try:
            index_folder = os.path.join(saved_folderpath, str(index_image))
            energy_filepath = os.path.join(index_folder, "energy_of_pixel.npy")
            np.load(energy_filepath)
            solid_angle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
            np.load(solid_angle_filepath)
        except FileNotFoundError:
            # Note: This is the longest part of the algorithm
            save_energy_and_solid_angle_matrix(index_image,folderpath=saved_folderpath,solid_angle_grid_width=3)

        geo_time = time.time()
        minutes, seconds = divmod(geo_time- start_time, 60)
        print(f"Geometry Initialisation runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

        # Has built in try
        spectrum_eng = Spectrum_Island(index_image,folderpath=saved_folderpath,bin_width=bin_width)
        spectrum_eng.energy_list_island(save=True)
        spectrum_eng.solid_angle_array()

    combined_spec_eng = Combine_Spectra(bin_width=bin_width,list_of_indices=list_image_indices,folderpath=saved_folderpath)
    combined_spec_eng.plot_average_normalised_corrected_spectra()


if __name__ == '__main__':
    main()

    pass
