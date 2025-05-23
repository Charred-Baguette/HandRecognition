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
    model_path = 'yolo11n.pt'
    last_checkpoint = 'runs/detect/hand_recognition/weights/last.pt'
    try:
        if os.path.exists(last_checkpoint):
            print(f"Found existing checkpoint at {last_checkpoint}")
            print("Attempting to resume training...")
            model = YOLO(last_checkpoint)
            results = model.train(
                data='dataset/data.yaml',
                epochs=500,
                imgsz=640,
                batch=64,
                name='hand_recognition',
                exist_ok=True,
                optimizer='auto',
                resume=True
            )
            
            # Add validation
            print("\nValidating final model...")
            val_results = model.val(data='dataset/data.yaml')
            print(f"\nValidation Results:")
            print(f"mAP50-95: {val_results.box.map}")
            print(f"mAP50: {val_results.box.map50}")
            print(f"mAP75: {val_results.box.map75}")
            print(f"Precision: {val_results.box.p}")
            print(f"Recall: {val_results.box.r}")

        else:
            print("Starting new training session...")
            model = YOLO(model_path)
            results = model.train(
                data='dataset/data.yaml',
                epochs=500,
                imgsz=640,
                batch=64,
                optimizer='auto',
                name='hand_recognition',
                exist_ok=True,
                resume=False  # Don't try to resume for new training
            )
            
            # Add validation
            print("\nValidating final model...")
            val_results = model.val(data='dataset/data.yaml')
            print(f"\nValidation Results:")
            print(f"mAP50-95: {val_results.box.map}")
            print(f"mAP50: {val_results.box.map50}")
            print(f"mAP75: {val_results.box.map75}")
            print(f"Precision: {val_results.box.p}")
            print(f"Recall: {val_results.box.r}")

    except AssertionError as e:
        print("Previous training was completed. Starting fresh training...")
        model = YOLO(model_path)
        results = model.train(
            data='dataset/data.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name='hand_recognition',
            exist_ok=True,
            resume=False
        )
        
        # Add validation
        print("\nValidating final model...")
        val_results = model.val(data='dataset/data.yaml')
        print(f"\nValidation Results:")
        print(f"mAP50-95: {val_results.box.map}")
        print(f"mAP50: {val_results.box.map50}")
        print(f"mAP75: {val_results.box.map75}")
        print(f"Precision: {val_results.box.p}")
        print(f"Recall: {val_results.box.r}")

if __name__ == "__main__":
    print("Training YOLO model...")
    print("Please wait...")
    train_yolo()