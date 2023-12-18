import zipfile
import os

def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all contents into the specified directory
        zip_ref.extractall(extract_to_path)


zip_file_path = '/gpfs/space/home/amlk/UBC-OCEAN.zip'
extract_to_path = '/gpfs/space/home/amlk/train_images'

# Make sure the destination folder exists, create if not
if not os.path.exists(extract_to_path):
    os.makedirs(extract_to_path)

unzip_file(zip_file_path, extract_to_path)