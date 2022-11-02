from msilib.schema import Directory
import os

dir = './'

for filename in os.listdir(dir):
    if filename.startswith("_"):
        os.rename(filename, filename[1:])