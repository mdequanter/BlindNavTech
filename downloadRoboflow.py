import roboflow
import os
import json

rf = roboflow.Roboflow()
project = rf.workspace().project("personalobjects")

#model = project.version(1).download("yolov5")

# Download een versie die segmentatie ondersteunt
model = project.version(1).download("coco-segmentation")

# COCO dataset pad
coco_json_path = "PersonalObjects-1\\train\\_annotations.coco.json"
image_dir = "PersonalObjects-1\\train\\images"
output_label_dir = "PersonalObjects-1\\train\\labels"

# Maak de output map als deze nog niet bestaat
os.makedirs(output_label_dir, exist_ok=True)

# Laad de COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Mapping van categorieën
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Verwerk alle annotaties
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
    
    image_path = os.path.join(image_dir, image_info["file_name"])
    height, width = image_info["height"], image_info["width"]
    
    # Haal de segmentatiepolygonen op
    segmentation = annotation["segmentation"]
    category_id = annotation["category_id"]
    
    # YOLOv5-Seg labelbestand maken
    new_filename = image_info['file_name'].replace('.jpg', '.txt') 
    label_filename = os.path.join(output_label_dir, new_filename)
    
    with open(label_filename, "w") as f:
        for polygon in segmentation:
            # Normaliseer polygon-coördinaten (x/w, y/h)
            normalized_polygon = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] / width
                y = polygon[i + 1] / height
                normalized_polygon.append(f"{x} {y}")

            # YOLOv5-Seg format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
            f.write(f"{category_id} " + " ".join(normalized_polygon) + "\n")

print("Conversie voltooid! YOLOv5-Seg labels zijn opgeslagen in:", output_label_dir)
