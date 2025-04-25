#setup script
def setup():
    print("Importing dependencies...")
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
   
    print("Dependencies imported successfully!")
    
    dataset_path = "Type_01_(Raw_Gesture)"
    if os.path.exists(dataset_path):
        print(f"Dataset directory '{dataset_path}' already exists.")

    else: 
        print("Please get dataset...")

    

if __name__ == "__main__":
    setup()
    print("Setup complete!")
    print("You can now run the main script.")