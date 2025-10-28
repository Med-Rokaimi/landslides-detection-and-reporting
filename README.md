# landslides-detection-and-reporting
Implementation of the paper: From Landslide Detection to Multi-Source LLM-Based Reporting: A Complete Framework for Rapid Assessment of Post-Disaster Scenarios.


<img width="1951" height="2002" alt="framework" src="https://github.com/user-attachments/assets/5bb29dbe-7adb-43a4-bf4d-861438f5a1cc" />

**Contents:**
**yolo_trainin.py**:Fine-tuning YOLOv11 for landslide detection. The pretrained model takes RGB images as input and outputs the coordinates of detected landslides in YOLO format (normalized).
**mistral_caption.py**: Reading the CSV file (Ministral_inputs.csv), which contains the images, their generated coordinates for each landslide instance, and the corresponding geo-features already extracted from rasters. Then, additional metadata is calculated at the image level, including the number of landslides in the image, their overlap or clustering, and their areas. All this information is then fed into the Mistral LLM through a prompt to generate a report for each image.

