import os
import shutil

dataset_path= "C:\\Users\\RZAMBRAN\\.cache\\kagglehub\\datasets\\veeralakrishna\\butterfly-dataset\\versions\\1\\leedsbutterfly"

images_path = os.path.join(dataset_path, "images")
organized_path = os.path.join(dataset_path, "organized")

os.makedirs(organized_path, exist_ok=True)

for fname in os.listdir(images_path):
    if fname.endswith(".png"):
        class_id = fname[0:3]  # "001", "002", etc.
        class_folder = os.path.join(organized_path, class_id)
        os.makedirs(class_folder, exist_ok=True)
        shutil.copy(os.path.join(images_path, fname), os.path.join(class_folder, fname))

print("Reorganized dataset at:", organized_path)