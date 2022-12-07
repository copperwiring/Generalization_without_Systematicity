import os
import urllib.request
import zipfile

url = 'https://github.com/brendenlake/SCAN/archive/refs/heads/master.zip'

filename = "SCAN.zip"

print('From:', url)
print('  To:', filename)
destination_folder = "SCAN"

urllib.request.urlretrieve(url, filename)
print("Data folder fetched")
# ---

os.makedirs(destination_folder, exist_ok=True)

print('Please wait. Unzipping the file..')

zip_file = zipfile.ZipFile(filename)
zip_file.extractall(destination_folder)

print(f"Data download completed. Refresh the base directory and look for {destination_folder} directory")