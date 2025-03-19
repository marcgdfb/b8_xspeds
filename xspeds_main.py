from legacy_code.spectrum_engine_v4 import *


def main(saved_folderpath="stored_variables",use_saved_values=False,list_image_indices=list_data):
    """
    :param list_image_indices:
    :param saved_folderpath: Folderpath for where values are/will be saved
    :param use_saved_values: Use saved values
    :return:
    """
    print("main:")
    print("list_image_indices:", list_image_indices)
    print("Using saved values:", use_saved_values)

    if not use_saved_values:
        calibrate_quadratic(list_indices=list_image_indices, folderpath=saved_folderpath)
        calibrate_ellipse(list_indices=list_image_indices, folderpath=saved_folderpath)
        calibrate_geometric(list_indices=list_image_indices, folderpath=saved_folderpath)
        calibrate_energy_solidAngle(list_indices=list_image_indices, folderpath=saved_folderpath)
        for index_image in list_image_indices:
            solid_angle_per_energy_bin(index_image,bin_width=1,folderpath=saved_folderpath)
            spectrum_eng = Spectrum(indexOfInterest=index_image,folderpath=saved_folderpath)
            spectrum_eng.islandSpectrum_SolidAngle_with_uncertainty(bin_width=1,save=True)





if __name__ == '__main__':
    # main(list_image_indices=[11],)

    pass
