import os
import rarfile
from rarfile import BadRarFile
from dotenv import load_dotenv
load_dotenv()

data_path = os.getenv("DATA_PATH")
dataset_path = os.path.join(data_path, "archived")
extract_path = os.path.join(data_path, "raw")

# This script recursively walks through the folder
# and extracts all .rar files into a specified directory
try: 
    for (root, dir, files) in os.walk(dataset_path):
        for rar_path in [os.path.join(root, file) for file in files if file.endswith("rar")]:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(extract_path)
    
    print("Extraction complete!")
except BadRarFile as e:
    print(e, rar_path)

