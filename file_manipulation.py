import pandas as pd
from calibrate_geometry_v4 import *
import re

class FileManipulation:
    def __init__(self,list_data_indices=list_data,folderpath="stored_variables"):
        self.list_data_indices = list_data_indices
        self.folderpath = folderpath

    def ellipse_excel_append(self):
        excel_filename = "ellipse_params.xlsx"
        excel_filename_temp = "ellipse_params_temp.xlsx"
        excl_filepath = os.path.join(self.folderpath, excel_filename)

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

        for iOI in self.list_data_indices:
            try:
                left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(iOI,folderpath=self.folderpath)

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
            excl_filepath = os.path.join(self.folderpath, excel_filename_temp)
            df.to_excel(excl_filepath, index=False)

    def geometric_excel_append(self):

        excel_filename = "Geometry.xlsx"
        excel_filename_temp = "Geometry_temp.xlsx"
        excl_filepath = os.path.join(self.folderpath, excel_filename)

        dict_data = {
            "image_index": [],
            "crys_pitch": [],
            "crys_roll": [],
            "cam_pitch": [],
            "cam_roll": [],
            "r_cam": [],
        }
        for idx_data_ in self.list_data_indices:
            try:
                crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(idx_data_,self.folderpath)

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
            excl_filepath = os.path.join(self.folderpath, excel_filename_temp)
            df.to_excel(excl_filepath, index=False)


    def createLossExcel(self):

        vals_dict = {
            "Image Index": [],
            "Left Quad Loss": [],
            "Right Quad Loss": [],
            "Quad Fitting Time": [],
            "Left Ellipse Loss": [],
            "Right Ellipse Loss": [],
            "Ellipse Fitting Time": [],
        }


        def extract_params(filepath):
            # Read the file content
            with open(filepath, 'r') as file:
                log_text = file.read()

            # Extract all start times and take the most recent (last occurrence)
            start_times = re.findall(r"Start: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_text)
            if not start_times:
                raise ValueError("No start times found")
            start_time = datetime.strptime(start_times[-1], "%Y-%m-%d %H:%M:%S")

            # Extract all finish times and take the most recent (last occurrence)
            finish_times = re.findall(r"Finish: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_text)
            if not finish_times:
                raise ValueError("No finish times found")
            finish_time = datetime.strptime(finish_times[-1], "%Y-%m-%d %H:%M:%S")

            time_taken = (finish_time - start_time).total_seconds()

            # Extract all occurrences of "Loss = <number>"
            loss_matches = re.findall(r"Loss = ([\-\d\.]+)", log_text)
            if len(loss_matches) < 2:
                raise ValueError("Less than two Loss values found")

            # The penultimate value is for left, and the last is for right
            left_loss = float(loss_matches[-2])
            right_loss = float(loss_matches[-1])

            return left_loss, right_loss, time_taken

        for iOI in self.list_data_indices:
            try:
                index_folder = os.path.join(self.folderpath, str(iOI))
                log_quad_filepath = os.path.join(index_folder, "quadratic_fits_log.txt")
                log_ellipse_filepath = os.path.join(index_folder, "ellipse_fits_log.txt")

                quad_left_loss, quad_right_loss, quad_time_taken = extract_params(log_quad_filepath)
                ellipse_left_loss, ellipse_right_loss, ellipse_time_taken = extract_params(log_ellipse_filepath)

                vals_dict["Image Index"].append(iOI)
                vals_dict["Left Quad Loss"].append(quad_left_loss)
                vals_dict["Right Quad Loss"].append(quad_right_loss)
                vals_dict["Quad Fitting Time"].append(quad_time_taken)
                vals_dict["Left Ellipse Loss"].append(ellipse_left_loss)
                vals_dict["Right Ellipse Loss"].append(ellipse_right_loss)
                vals_dict["Ellipse Fitting Time"].append(ellipse_time_taken)

            except FileNotFoundError:
                print(f"File {iOI} not found.")
                continue

        df = pd.DataFrame(vals_dict)

        excel_filename = "loss_quad_ellipse.xlsx"
        excel_filename_temp = "loss_quad_ellipse_temp.xlsx"
        excl_filepath = os.path.join(self.folderpath, excel_filename)

        if not os.path.exists(excl_filepath):
            df.to_excel(excl_filepath, index=False)
        else:
            print("File already exists. Creating new one with changed name")
            excl_filepath = os.path.join(self.folderpath, excel_filename_temp)
            df.to_excel(excl_filepath, index=False)

    def plotLoss_quadVSelipse(self):

        excel_filename = "loss_quad_ellipse.xlsx"
        excl_filepath = os.path.join(self.folderpath, excel_filename)

        df = pd.read_excel(excl_filepath)

        image_indices = df["Image Index"].tolist()
        left_quad_losses = df["Left Quad Loss"].tolist()
        right_quad_losses = df["Right Quad Loss"].tolist()
        left_ellipse_loss = df["Left Ellipse Loss"].tolist()
        right_ellipse_loss = df["Right Ellipse Loss"].tolist()

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot left losses on the first graph
        axs[0].plot(image_indices, left_quad_losses, marker="o", label="Left Quad Loss")
        axs[0].plot(image_indices, left_ellipse_loss, marker="o", label="Left Ellipse Loss")
        axs[0].set_title("Left Losses vs Image Index")
        axs[0].set_xlabel("Image Index")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        # Plot right losses on the second graph
        axs[1].plot(image_indices, right_quad_losses, marker="o", label="Right Quad Loss")
        axs[1].plot(image_indices, right_ellipse_loss, marker="o", label="Right Ellipse Loss")
        axs[1].set_title("Right Losses vs Image Index")
        axs[1].set_xlabel("Image Index")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


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
    plt.ylabel("j index")
    plt.xlabel("i index")
    plt.show()

def plot_solidAngle(indexOI,folderpath="stored_variables"):
    index_folder = os.path.join(folderpath, str(indexOI))
    solidA_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")

    mat_SA = np.load(solidA_filepath)

    plt.imshow(mat_SA, cmap='gray')
    plt.title(f"solid angle for image {indexOI}")
    plt.colorbar(label="Solid Angle")
    plt.show()



if __name__ == "__main__":

    # FileManipulation().plotLoss_quadVSelipse()

    # plot_solidAngle(8)

    # geometric_excel_append()
    # ellipse_excel_append()

    pass
