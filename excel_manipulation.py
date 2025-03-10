import pandas as pd
import numpy as np
from imagePreProcessing import list_data

class Excel:
    def __init__(self, file_path, sheetname=None):
        self.file_path = file_path

        if sheetname is not None:
            self.sheetname = sheetname
            self.df = pd.read_excel(self.file_path, self.sheetname)
        else:
            self.df = pd.read_excel(self.file_path)


def abc_excel_append():
    cols = ['image_index', 'A_left', 'B_left', 'C_left', 'A_right', 'B_right',
            'C_right', 'outlier']

    excl_filename = r"/stored_variables/ABC_lines.xlsx"

    exc = Excel(excl_filename)
    df = exc.df.copy()

    for idx_data in list_data:

        folderpath_quad = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised_2"
        filename = f"{idx_data}"

        vals = np.load(f"{folderpath_quad}/{filename}.npy")
        Aleft = vals[0, 0]
        Bleft = vals[0, 1]
        Cleft = vals[0, 2]
        Aright = vals[1, 0]
        Bright = vals[1, 1]
        Cright = vals[1, 2]

        rowAppend = [idx_data, Aleft, Bleft, Cleft, Aright, Bright, Cright,0]
        new_row_df = pd.DataFrame([rowAppend], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_excel(excl_filename, index=False)

def geometric_append():

    excl_filename = r"/stored_variables/Geometry.xlsx"

    exc = Excel(excl_filename)
    df = exc.df.copy()

    for idx_data in list_data:

        folderpath_geo = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\geometry"
        filename = f"{idx_data}"
        optimised_geoParams = np.load(f"{folderpath_geo}/{filename}.npy")
        crysPitch = optimised_geoParams[0]
        CrysRoll = optimised_geoParams[1]
        CamPitch = optimised_geoParams[2]
        CamRoll = optimised_geoParams[3]
        rcam = optimised_geoParams[4]

        rowAppend = [idx_data, crysPitch, CrysRoll, CamPitch,CamRoll, rcam,0]
        new_row_df = pd.DataFrame([rowAppend], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_excel(excl_filename, index=False)


