#setup script
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
import shutil
import yaml
print("Dependencies imported successfully!")
def setup():    
    dataset_path = "Type_01_(Raw_Gesture)"
    if os.path.exists(dataset_path):
        print(f"Dataset directory '{dataset_path}' already exists.")

    else: 
        print("Please get dataset...")
        
def prepare_dataset():
    # Define paths
    dataset_root = "Type_01_(Raw_Gesture)"
    yolo_dataset = "dataset"
    
    # Create YOLO format directories
    os.makedirs(f"{yolo_dataset}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_dataset}/images/val", exist_ok=True)
    os.makedirs(f"{yolo_dataset}/labels/train", exist_ok=True)
    os.makedirs(f"{yolo_dataset}/labels/val", exist_ok=True)
    
    # Create class mapping
    classes = sorted([d for d in os.listdir(dataset_root) 
                     if os.path.isdir(os.path.join(dataset_root, d))])
    
    # Create data.yaml with absolute paths
    current_dir = os.path.abspath(os.getcwd())
    data = {
        'train': os.path.join(current_dir, 'dataset/images/train'),
        'val': os.path.join(current_dir, 'dataset/images/val'),
        'nc': len(classes),
        'names': classes
    }
    
    with open(f"{yolo_dataset}/data.yaml", 'w') as f:
        yaml.dump(data, f)
    
    # Process each class folder
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_root, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        # Split 80/20 for train/val
        split = int(len(images) * 0.8)
        train_images = images[:split]
        val_images = images[split:]
        
        # Process training images
        for img_name in train_images:
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            w, h = img.size
            
            # Copy image
            shutil.copy2(img_path, f"{yolo_dataset}/images/train/{img_name}")
            
            # Create YOLO format label
            with open(f"{yolo_dataset}/labels/train/{img_name.replace('.jpg', '.txt')}", 'w') as f:
                # Format: class_id x_center y_center width height
                f.write(f"{idx} 0.5 0.5 1.0 1.0\n")
        
        # Process validation images
        for img_name in val_images:
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            w, h = img.size
            
            # Copy image
            shutil.copy2(img_path, f"{yolo_dataset}/images/val/{img_name}")
            
            # Create YOLO format label
            with open(f"{yolo_dataset}/labels/val/{img_name.replace('.jpg', '.txt')}", 'w') as f:
                f.write(f"{idx} 0.5 0.5 1.0 1.0\n")
    

if __name__ == "__main__":
    setup()
    print("Setup complete!")
    print("You can now run the main script.")