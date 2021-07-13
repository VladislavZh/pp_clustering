import os


def create_folder(path_to_folder: str, rewrite: bool = False):
    """
        creates a folder, if rewrite, then clears the folder if it exists
    """
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        if not rewrite:
            return False
        clear_folder(path_to_folder)
        return True
    os.mkdir(path_to_folder)


def clear_folder(path_to_folder: str):
    """
        clears the folder if exists
    """
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        for file in os.listdir(path_to_folder):
            os.remove(path_to_folder + '/' + file)
        return True
    return False
