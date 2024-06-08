from config import *
from utils import *

def main():
    # download_save_numpy(DATA_LINKS, ZIP_DATA_PATH, NUMPY_DATA_PATH)
    combine_and_write_numpy(NUMPY_DATA_PATH, NUMPY_COMBINE_PATH)


if __name__ == "__main__":
    main()