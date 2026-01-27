"""
Script to generate example/simulated data files for each task type.
Run this script to create sample data files in the data/examples directory.
"""

import numpy as np
import pandas as pd
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "examples")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_classification_data():
    """Generate binary classification example data."""
    n_samples = 500

    # Simulate disease presence (30% positive rate)
    y_true = np.random.binomial(1, 0.3, n_samples)

    # Simulate model predictions with reasonable AUC (~0.85)
    y_pred = np.zeros(n_samples)
    y_pred[y_true == 1] = np.random.beta(5, 2, sum(y_true == 1))  # Higher scores for positives
    y_pred[y_true == 0] = np.random.beta(2, 5, sum(y_true == 0))  # Lower scores for negatives

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })

    filepath = os.path.join(OUTPUT_DIR, "classification_example.csv")
    df.to_csv(filepath, index=False)
    print(f"Created: {filepath}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Positive rate: {y_true.mean():.1%}")
    return filepath


def generate_regression_data():
    """Generate regression example data (e.g., tumor size prediction)."""
    n_samples = 300

    # Simulate true tumor sizes (mm) - log-normal distribution
    y_true = np.random.lognormal(mean=2.5, sigma=0.5, size=n_samples)
    y_true = np.clip(y_true, 5, 100)  # Realistic range: 5-100mm

    # Simulate predictions with some error
    noise = np.random.normal(0, 3, n_samples)  # ~3mm average error
    y_pred = y_true + noise
    y_pred = np.clip(y_pred, 0, 120)  # Keep predictions reasonable

    df = pd.DataFrame({
        'y_true': np.round(y_true, 2),
        'y_pred': np.round(y_pred, 2)
    })

    filepath = os.path.join(OUTPUT_DIR, "regression_example.csv")
    df.to_csv(filepath, index=False)
    print(f"Created: {filepath}")
    print(f"  - Samples: {n_samples}")
    print(f"  - True value range: [{y_true.min():.1f}, {y_true.max():.1f}]")
    return filepath


def generate_segmentation_data_2d():
    """Generate 2D segmentation example data (e.g., lung nodule segmentation)."""
    height, width = 256, 256

    # Create ground truth mask with a circular lesion
    y_true = np.zeros((height, width), dtype=np.uint8)
    center_y, center_x = 128, 128
    radius = 40

    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    y_true[mask] = 1

    # Create prediction with slight offset and size difference (simulating model output)
    y_pred = np.zeros((height, width), dtype=np.uint8)
    pred_center_y, pred_center_x = 130, 132  # Slight offset
    pred_radius = 38  # Slightly smaller

    mask_pred = (x - pred_center_x)**2 + (y - pred_center_y)**2 <= pred_radius**2
    y_pred[mask_pred] = 1

    # Add some noise to prediction (small FP regions)
    noise_mask = np.random.random((height, width)) < 0.001
    y_pred[noise_mask] = 1

    gt_path = os.path.join(OUTPUT_DIR, "segmentation_2d_ground_truth.npy")
    pred_path = os.path.join(OUTPUT_DIR, "segmentation_2d_prediction.npy")

    np.save(gt_path, y_true)
    np.save(pred_path, y_pred)

    print(f"Created: {gt_path}")
    print(f"Created: {pred_path}")
    print(f"  - Shape: {y_true.shape}")
    print(f"  - GT pixels: {y_true.sum()}, Pred pixels: {y_pred.sum()}")
    return gt_path, pred_path


def generate_segmentation_data_3d():
    """Generate 3D segmentation example data (e.g., liver segmentation from CT)."""
    depth, height, width = 32, 128, 128

    # Create ground truth mask with an ellipsoid organ
    y_true = np.zeros((depth, height, width), dtype=np.uint8)

    center_z, center_y, center_x = 16, 64, 64
    radius_z, radius_y, radius_x = 10, 30, 35

    z, y, x = np.ogrid[:depth, :height, :width]
    mask = ((x - center_x)/radius_x)**2 + ((y - center_y)/radius_y)**2 + ((z - center_z)/radius_z)**2 <= 1
    y_true[mask] = 1

    # Create prediction with slight differences
    y_pred = np.zeros((depth, height, width), dtype=np.uint8)

    pred_center_z, pred_center_y, pred_center_x = 16, 65, 63
    pred_radius_z, pred_radius_y, pred_radius_x = 9, 28, 33

    mask_pred = ((x - pred_center_x)/pred_radius_x)**2 + ((y - pred_center_y)/pred_radius_y)**2 + ((z - pred_center_z)/pred_radius_z)**2 <= 1
    y_pred[mask_pred] = 1

    gt_path = os.path.join(OUTPUT_DIR, "segmentation_3d_ground_truth.npy")
    pred_path = os.path.join(OUTPUT_DIR, "segmentation_3d_prediction.npy")

    np.save(gt_path, y_true)
    np.save(pred_path, y_pred)

    print(f"Created: {gt_path}")
    print(f"Created: {pred_path}")
    print(f"  - Shape: {y_true.shape}")
    print(f"  - GT voxels: {y_true.sum()}, Pred voxels: {y_pred.sum()}")
    return gt_path, pred_path


def generate_multiclass_segmentation_data():
    """Generate multi-class segmentation data (e.g., brain tumor segmentation)."""
    height, width = 256, 256

    # Create ground truth with multiple classes
    # 0 = background, 1 = tumor core, 2 = edema, 3 = enhancing tumor
    y_true = np.zeros((height, width), dtype=np.uint8)

    y, x = np.ogrid[:height, :width]

    # Edema (largest region) - class 2
    edema_mask = (x - 128)**2 + (y - 128)**2 <= 50**2
    y_true[edema_mask] = 2

    # Tumor core - class 1
    core_mask = (x - 128)**2 + (y - 128)**2 <= 30**2
    y_true[core_mask] = 1

    # Enhancing tumor (innermost) - class 3
    enhancing_mask = (x - 128)**2 + (y - 128)**2 <= 15**2
    y_true[enhancing_mask] = 3

    # Create prediction with some errors
    y_pred = np.zeros((height, width), dtype=np.uint8)

    # Slightly different boundaries
    edema_mask_pred = (x - 130)**2 + (y - 126)**2 <= 48**2
    y_pred[edema_mask_pred] = 2

    core_mask_pred = (x - 130)**2 + (y - 126)**2 <= 28**2
    y_pred[core_mask_pred] = 1

    enhancing_mask_pred = (x - 130)**2 + (y - 126)**2 <= 14**2
    y_pred[enhancing_mask_pred] = 3

    gt_path = os.path.join(OUTPUT_DIR, "segmentation_multiclass_ground_truth.npy")
    pred_path = os.path.join(OUTPUT_DIR, "segmentation_multiclass_prediction.npy")

    np.save(gt_path, y_true)
    np.save(pred_path, y_pred)

    print(f"Created: {gt_path}")
    print(f"Created: {pred_path}")
    print(f"  - Shape: {y_true.shape}")
    print(f"  - Classes: 0=background, 1=tumor core, 2=edema, 3=enhancing")
    return gt_path, pred_path


def generate_detection_data():
    """Generate object detection example data (e.g., lung nodule detection)."""
    n_images = 50

    predictions = []
    ground_truths = []

    for img_idx in range(n_images):
        # Random number of ground truth nodules (0-4 per image)
        n_nodules = np.random.choice([0, 1, 1, 2, 2, 3, 4], p=[0.1, 0.25, 0.25, 0.2, 0.1, 0.07, 0.03])

        img_gt = []
        img_pred = []

        for _ in range(n_nodules):
            # Generate ground truth box
            x1 = int(np.random.randint(50, 400))
            y1 = int(np.random.randint(50, 400))
            size = int(np.random.randint(20, 80))
            gt_box = [x1, y1, x1 + size, y1 + size]
            img_gt.append(gt_box)

            # 80% chance of detecting this nodule
            if np.random.random() < 0.8:
                # Add some localization error
                offset = np.random.randint(-8, 8, 4)
                pred_box = [
                    int(max(0, gt_box[0] + offset[0])),
                    int(max(0, gt_box[1] + offset[1])),
                    int(gt_box[2] + offset[2]),
                    int(gt_box[3] + offset[3])
                ]
                score = float(np.random.uniform(0.5, 0.98))
                img_pred.append({"box": pred_box, "score": round(score, 3)})

        # Add some false positives (0-2 per image)
        n_fp = np.random.choice([0, 0, 1, 1, 2], p=[0.4, 0.2, 0.2, 0.15, 0.05])
        for _ in range(n_fp):
            x1 = int(np.random.randint(50, 400))
            y1 = int(np.random.randint(50, 400))
            size = int(np.random.randint(15, 50))
            fp_box = [x1, y1, x1 + size, y1 + size]
            score = float(np.random.uniform(0.3, 0.7))  # Lower confidence for FPs
            img_pred.append({"box": fp_box, "score": round(score, 3)})

        predictions.append(img_pred)
        ground_truths.append(img_gt)

    data = {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "metadata": {
            "description": "Simulated lung nodule detection data",
            "n_images": n_images,
            "box_format": "[x1, y1, x2, y2]"
        }
    }

    filepath = os.path.join(OUTPUT_DIR, "detection_example.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    total_gt = sum(len(gt) for gt in ground_truths)
    total_pred = sum(len(pred) for pred in predictions)

    print(f"Created: {filepath}")
    print(f"  - Images: {n_images}")
    print(f"  - Total ground truth boxes: {total_gt}")
    print(f"  - Total predictions: {total_pred}")
    return filepath


def main():
    print("=" * 60)
    print("Generating example data files for Omnibin")
    print("=" * 60)
    print()

    print("1. Binary Classification Data")
    print("-" * 40)
    generate_classification_data()
    print()

    print("2. Regression Data")
    print("-" * 40)
    generate_regression_data()
    print()

    print("3. 2D Segmentation Data")
    print("-" * 40)
    generate_segmentation_data_2d()
    print()

    print("4. 3D Segmentation Data")
    print("-" * 40)
    generate_segmentation_data_3d()
    print()

    print("5. Multi-class Segmentation Data")
    print("-" * 40)
    generate_multiclass_segmentation_data()
    print()

    print("6. Object Detection Data")
    print("-" * 40)
    generate_detection_data()
    print()

    print("=" * 60)
    print("All example data files generated successfully!")
    print(f"Location: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
