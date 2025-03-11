from spc_engine_v4 import *


def main(saved_folderpath=None):

    if saved_folderpath is None:
        saved_folderpath = "stored_variables"
        files = os.listdir(saved_folderpath)  # List files in the folder
        print("Files in folder:", files)




if __name__ == '__main__':
    main()
