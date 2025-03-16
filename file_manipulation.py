import pandas as pd
from calibrate_geometry_v4 import *

class Excel:
    def __init__(self, file_path, sheetname=None):
        self.file_path = file_path

        if sheetname is not None:
            self.sheetname = sheetname
            self.df = pd.read_excel(self.file_path, self.sheetname)
        else:
            self.df = pd.read_excel(self.file_path)


def ellipse_excel_append(list_data_indices=list_data,folderpath="stored_variables"):
    excel_filename = "ellipse_params.xlsx"
    excel_filename_temp = "ellipse_params_temp.xlsx"
    excl_filepath = os.path.join(folderpath, excel_filename)

    vals_dict = {
        "image_index": [],
        "y0 left": [],
        "A left": [],
        "B left": [],
        "C left": [],
        "y0 right": [],
        "A right": [],
        "B right": [],
        "C right": [],
    }

    for iOI in list_data_indices:
        try:
            left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(iOI,folderpath=folderpath)

            vals_dict["image_index"].append(iOI)
            vals_dict["y0 left"].append(left_vars_y0ABc[0])
            vals_dict["A left"].append(left_vars_y0ABc[1])
            vals_dict["B left"].append(left_vars_y0ABc[2])
            vals_dict["C left"].append(left_vars_y0ABc[3])

            vals_dict["y0 right"].append(right_vars_y0ABc[0])
            vals_dict["A right"].append(right_vars_y0ABc[1])
            vals_dict["B right"].append(right_vars_y0ABc[2])
            vals_dict["C right"].append(right_vars_y0ABc[3])
        except FileNotFoundError:
            print(f"File {iOI} not found.")
            continue

    df = pd.DataFrame(vals_dict)

    if not os.path.exists(excl_filepath):
        df.to_excel(excl_filepath, index=False)
    else:
        print("File already exists. Creating new one with changed name")
        excl_filepath = os.path.join(folderpath, excel_filename_temp)
        df.to_excel(excl_filepath, index=False)


def geometric_excel_append(list_data_indices=list_data,folderpath="stored_variables"):

    excel_filename = "Geometry.xlsx"
    excel_filename_temp = "Geometry_temp.xlsx"
    excl_filepath = os.path.join(folderpath, excel_filename)

    dict_data = {
        "image_index": [],
        "crys_pitch": [],
        "crys_roll": [],
        "cam_pitch": [],
        "cam_roll": [],
        "r_cam": [],
    }
    for idx_data_ in list_data_indices:
        try:
            crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(idx_data_,folderpath)

            dict_data["image_index"].append(idx_data_)
            dict_data["crys_pitch"].append(crys_pitch)
            dict_data["crys_roll"].append(crys_roll)
            dict_data["cam_pitch"].append(cam_pitch)
            dict_data["cam_roll"].append(cam_roll)
            dict_data["r_cam"].append(r_cam)
        except FileNotFoundError:
            print(f"File {idx_data_} not found.")
            continue

    df = pd.DataFrame(dict_data)
    if not os.path.exists(excl_filepath):
        df.to_excel(excl_filepath, index=False)
    else:
        print("File already exists. Creating new one with changed name")
        excl_filepath = os.path.join(folderpath, excel_filename_temp)
        df.to_excel(excl_filepath, index=False)


def save_recent_logged_ellipse(iOI,folderpath="stored_variables"):
    # initialise subfolder if it doesn't exist
    index_folder = os.path.join(folderpath, str(iOI))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    filepath = os.path.join(index_folder, "ellipse_fits.npy")
    left_all = [871.487462474521, 1096.831350742212, 2760.608277556325, 1273.7561661323955 ,0.4162190881801597]
    right_all = [847.5915575658971, 1496.5042331304744, 3392.599028828548, 1418.3171636061127 ,0.36931537313532725]
    ellipse_vars = np.array([
        left_all,
        right_all
    ])
    np.save(filepath, ellipse_vars)


# TODO: Compare matrices to see if there is a non neglibile difference between energy mappings:


def compare_energy_matrices(index1,index2,folderpath="stored_variables"):
    index_folder1 = os.path.join(folderpath, str(index1))
    index_folder2 = os.path.join(folderpath, str(index2))
    energy_filepath1 = os.path.join(index_folder1, "energy_of_pixel.npy")
    energy_filepath2 = os.path.join(index_folder2, "energy_of_pixel.npy")

    energy_mat_1 = np.load(energy_filepath1)
    energy_mat_2 = np.load(energy_filepath2)

    mat_dif = energy_mat_1 - energy_mat_2
    plt.imshow(mat_dif,cmap='gray')
    plt.title(f"Energy Gradient difference for image {index1} - {index2}")
    plt.colorbar(label="Energy Difference (eV)")
    plt.show()

def plot_solidAngle(indexOI,folderpath="stored_variables"):
    index_folder = os.path.join(folderpath, str(indexOI))
    solidA_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")

    mat_SA = np.load(solidA_filepath)

    plt.imshow(mat_SA, cmap='gray')
    plt.title(f"solid angle for image {indexOI}")
    plt.colorbar(label="Solid Angle")
    plt.show()


plot_solidAngle(8)

# compare_energy_matrices(7,8)

# geometric_excel_append()
# ellipse_excel_append()

