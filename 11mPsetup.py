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
    classification_dataset = "classification_dataset"
    
    # Verify source dataset exists
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset directory '{dataset_root}' not found")
    
    # Clean existing classification dataset if exists
    if os.path.exists(classification_dataset):
        shutil.rmtree(classification_dataset)
        print(f"Cleaned existing {classification_dataset} directory")
    
    # Create classification format directories
    splits = {'train': 0.8, 'val': 0.2}
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]  # A to Z
    
    # Create directory structure
    for split in splits:
        for letter in letters:
            letter_dir = os.path.join(classification_dataset, split, letter)
            os.makedirs(letter_dir, exist_ok=True)
            print(f"Created directory: {letter_dir}")
    
    total_images = 0
    processed_classes = set()
    
    # Process each folder in dataset root
    for folder_name in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Get all valid images from the folder
        images = [f for f in os.listdir(folder_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Group images by their first letter
        for img_name in images:
            if not img_name[0].isalpha():
                continue
                
            letter = img_name[0].upper()
            if letter not in letters:
                continue
                
            # Determine split (train/val) using random selection
            split = 'train' if np.random.random() < splits['train'] else 'val'
            
            # Copy image to appropriate folder
            src_path = os.path.join(folder_path, img_name)
            dst_path = os.path.join(classification_dataset, split, letter, img_name)
            
            try:
                shutil.copy2(src_path, dst_path)
                total_images += 1
                processed_classes.add(letter)
                print(f"Copied {img_name} to {split}/{letter}/")
            except Exception as e:
                print(f"Error copying {img_name}: {str(e)}")
    
    # Verify dataset creation
    if total_images == 0:
        raise RuntimeError("No images were copied to the classification dataset")
    
    # Print detailed statistics
    print(f"\nDataset prepared successfully:")
    print(f"- Total images: {total_images}")
    print(f"- Processed classes: {sorted(list(processed_classes))}")
    
    # Print statistics for each split
    for split in splits:
        split_path = os.path.join(classification_dataset, split)
        split_classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        class_counts = {}
        
        for cls in split_classes:
            class_path = os.path.join(split_path, cls)
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[cls] = count
        
        total_split_images = sum(class_counts.values())
        print(f"\n{split.capitalize()} set:")
        print(f"- Total images: {total_split_images}")
        print(f"- Classes: {len(split_classes)}")
        print("- Images per class:")
        for cls in sorted(class_counts.keys()):
            print(f"  {cls}: {class_counts[cls]}")

if __name__ == "__main__":
    setup()
    prepare_dataset()
    print("Setup complete!")