import cv2
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Paths
image_folder = 'images'
annotation_folder = 'labels/'



# select the dataset to use
dataset_samples = "100"


image_dir = Path(image_folder)
annotation_dir = Path(annotation_folder)

model_name = "runs/detect/train/weights/best.pt"

model_path = Path(model_name)
output_dir = Path(f"./inferance_all_test_{dataset_samples}/")
output_dir.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = 640  # YOLO input size
ORIGINAL_SIZE = 64  # original image size

# Function to load YOLO annotations and convert to supervision.Detections
def read_annotation(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Warning: Couldn't read image {image_path}")
        exit()
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    annotations = []
    image_name =  os.path.basename(image_path)
    img_height, img_width = image.shape[:2]
    annotation_file = os.path.join(annotation_dir, image_name.replace(".tif", ".txt"))
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            annotations.append({
                "bbox": [x_min, y_min, width, height],
                "category_id": class_id
            })
    return image, annotations

def from_custom_annotations(annotation_list: list) -> sv.Detections:
    xyxy, class_id = [], []
    for ann in annotation_list:
        x, y, w, h = ann["bbox"]
        xyxy.append([x, y, x + w, y + h])
        class_id.append(ann["category_id"])
    return sv.Detections(xyxy=np.array(xyxy), class_id=np.array(class_id))



# Load model once
model = YOLO(model_path)
conf_threshold = 0.3
save_samples = False
# Annotators
box_annotator = sv.BoxAnnotator()
#label_annotator = sv.LabelAnnotator()

# Process all images in the directory
image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() == ".tif"]
print(f"Found {len(image_paths)} images in {image_dir}")
outputs =[]
for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
    image, ground_annotations = read_annotation (image_path)
    results = model(image, verbose=False, conf=conf_threshold)
    detections = sv.Detections.from_ultralytics(results[0])
    boxes_640 = detections.xyxy

    boxes_64 = []
    # sclae coords to 64x64
    for box in boxes_640:
        x1, y1, x2, y2 = box
        # Rescale each coordinate to 64x64
        x1 = x1 * ORIGINAL_SIZE / INPUT_SIZE
        y1 = y1 * ORIGINAL_SIZE / INPUT_SIZE
        x2 = x2 * ORIGINAL_SIZE / INPUT_SIZE
        y2 = y2 * ORIGINAL_SIZE / INPUT_SIZE
        boxes_64.append([x1, y1, x2, y2])

    # Annotate image
    annotated = image.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)


    detections_gt = from_custom_annotations(ground_annotations)
   # Annotate ground truth
    frame_gt = box_annotator.annotate(scene=image.copy(), detections=detections_gt)

    
    # Save result
    if save_samples:
        output_path = output_dir / f"{image_path.stem}.jpg"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))#print(f"[{idx+1}/{len(image_paths)}] Saved: {output_path.name}")
        # save ground truth and predictions side by side
        axes[0].imshow(frame_gt)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(annotated)
        axes[1].set_title("Predictions")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_path)  # Save to file
        plt.close()  # Close the figure to free memory

    outputs.append({
            'image_id': os.path.basename(image_path),
            'boxes640':  boxes_640.tolist(),
            'boxes64' : boxes_64
        })

df = pd.DataFrame(outputs)
df.to_csv(f"predictions_yolo_{dataset_samples}.csv", index=False)


    
