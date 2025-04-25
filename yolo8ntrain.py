from libs.Camera import Camera
import Psetup
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
import shutil
import yaml
print("Running setup...")
Psetup.setup()
Psetup.prepare_dataset()
print("Setup complete!")


def check_dataset_structure():
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    base_path = 'dataset'
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            print(f"Created missing directory: {full_path}")

def train_yolo():
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
    
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')
    
    # Ensure dataset structure exists
    check_dataset_structure()
    
    # Train the model
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='hand_recognition',
        exist_ok=True  # Allow overwriting existing results
    )

if __name__ == "__main__":
    print("Training YOLO model...")
    print("Please wait...")
    train_yolo()