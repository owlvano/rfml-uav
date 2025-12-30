import os
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv("DATA_PATH")
load_path = os.path.join(data_path, "raw")

# This scripts flattens the file structure
# (moves all the files from the subdirectories in the root directory)
for (root, _, files) in os.walk(data_path):
    for file in files:
        os.rename(os.path.join(root, file), os.path.join(data_path, file))

# Remove all directories since they are empty
for (root, directories, _) in os.walk(data_path):
    for dir in directories:
        os.rmdir(os.path.join(root, dir))