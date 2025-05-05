from ultralytics import YOLO
import Psetup

print("Running setup...")
Psetup.setup()
print("Setup complete!")

def validate_yolo():
    best_weights = '11nCbest.pt'
    model = YOLO(best_weights)
    
    print("\nValidating model...")
    val_results = model.val(data='classification_dataset')
    print(f"\nValidation Results:")
    print(f"Accuracy: {val_results.top1}")
    print(f"Top-5 Accuracy: {val_results.top5}")

if __name__ == "__main__":
    print("Validating YOLO model...")
    validate_yolo()