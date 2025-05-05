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
    # Create classification dataset structure
    base_path = 'classification_dataset'
    splits = ['train', 'val']
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]  # A to Z
    
    # Create directory structure
    for split in splits:
        for letter in letters:
            full_path = os.path.join(base_path, split, letter)
            if not os.path.exists(full_path):
                os.makedirs(full_path, exist_ok=True)
                print(f"Created directory: {full_path}")

def train_yolo():
    model_path = 'yolo11n-cls.pt'  # Changed to v8 classifier
    last_checkpoint = 'runs/classify/hand_recognitionN11/weights/last.pt'
    try:
        check_dataset_structure()  # Create proper structure
        
        if os.path.exists(last_checkpoint):
            print(f"Found existing checkpoint at {last_checkpoint}")
            print("Attempting to resume training...")
            model = YOLO(last_checkpoint)
            results = model.train(
                data='classification_dataset',  # Changed path
                epochs=500,
                imgsz=640,
                batch=16,
                name='hand_recognitionN11',
                exist_ok=True,
                optimizer='auto',
                task='classify',  # Added task
                resume=True
            )
            
            # Updated validation metrics for classification
            print("\nValidating final model...")
            val_results = model.val()
            print(f"\nValidation Results:")
            print(f"Accuracy: {val_results.top1}")
            print(f"Top-5 Accuracy: {val_results.top5}")

        else:
            print("Starting new training session...")
            model = YOLO(model_path)
            results = model.train(
                data='classification_dataset',
                epochs=500,
                imgsz=640,
                batch=16,
                optimizer='auto',
                name='hand_recognitionN11',
                exist_ok=True,
                task='classify',
                resume=False
            )
            
            # Add validation
            print("\nValidating final model...")
            val_results = model.val()
            print(f"\nValidation Results:")
            print(f"Accuracy: {val_results.top1}")
            print(f"Top-5 Accuracy: {val_results.top5}")

    except AssertionError as e:
        print("Previous training was completed. Starting fresh training...")
        model = YOLO(model_path)
        results = model.train(
            data='classification_dataset',
            epochs=500,
            imgsz=640,
            batch=16,
            name='hand_recognitionN11',
            exist_ok=True,
            resume=False
        )
        
        print("\nValidating final model...")
        val_results = model.val()
        print(f"\nValidation Results:")
        print(f"Accuracy: {val_results.top1}")
        print(f"Top-5 Accuracy: {val_results.top5}")
if __name__ == "__main__":
    print("Training YOLO model...")
    print("Please wait...")
    train_yolo()