import sys
import logging
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from helpers.model_utils import parse_model_path, load_model, get_device
from helpers.data_utils import get_dataloaders


def test_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device
        ) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
    logging.info("Starting model evaluation on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predictions = outputs.max(1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = predictions[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                        
            except Exception as e:
                logging.error(f"Error during batch processing: {e}")
                continue
    
    overall_accuracy = correct / total if total > 0 else 0
    class_names = {0: "normal", 1: "pneumonia", 2: "tuberculosis"}
    class_accuracies = {}
    
    for class_id, class_name in class_names.items():
        if class_total[class_id] > 0:
            class_accuracies[class_name] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_name] = 0.0
    
    metrics = {
        "overall_accuracy": overall_accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "class_accuracies": class_accuracies,
        "class_totals": {class_names[k]: v for k, v in class_total.items()}
    }
    
    return metrics

def print_metrics(metrics):
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_samples']}")
    
    print("\nPer-Class Accuracy:")
    print("-" * 60)
    for class_name, accuracy in metrics['class_accuracies'].items():
        total = metrics['class_totals'][class_name]
        correct = int(accuracy * total)
        print(f"  {class_name.capitalize():15s}: {accuracy:.4f} ({accuracy*100:.2f}%) - {correct}/{total}")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] : %(message)s')
    
    try:
        model_path = parse_model_path(sys.argv)
        if model_path is None:
            logging.error("Model path is required for testing")
            sys.exit(1)
        
        device = get_device()
        model = load_model(model_path, device, num_classes=3)
        _, _, test_loader = get_dataloaders(batch_size=16)
        metrics = test_model(model, test_loader, device)
        
        print_metrics(metrics)
        logging.info("Testing completed successfully")
        
    except Exception as e:
        logging.error(f"Testing failed: {e}")
        sys.exit(1)
