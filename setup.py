#setup script
def setup():
    print("Installing dependencies...")
    import pandas as pd
    import os
    import torch
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import supervision as sv
    from PIL import Image
    import sys
    import zipfile
    import requests
   
    print("Dependencies installed successfully!")
    print("Downloading dataset...")
    
    url = "syntaxes.org/tmp/asl.zip"
    response = requests.get(url, stream=True)
    zip_path = "dataset.zip"
    
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("dataset")
    
    # Clean up zip file
    os.remove(zip_path)
    print("Dataset setup complete!")

if __name__ == "__main__":
    setup()
    print("Setup complete!")
    print("You can now run the main script.")