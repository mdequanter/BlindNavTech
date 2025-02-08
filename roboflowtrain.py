import tensorflow as tf
import pandas as pd
import os
import cv2
import numpy as np

# Paths to the dataset folders
DATASET_PATH = "netrasahaya-1"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Load class names (if predefined)
class_names = []

def load_annotations(csv_path):
    """Load annotations from the given CSV file."""
    df = pd.read_csv(csv_path)
    return df

def parse_annotations(df, image_folder):
    """Group bounding boxes by filename."""
    images = {}
    for _, row in df.iterrows():
        filename = row['filename']
        filepath = os.path.join(image_folder, filename)
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        label = row['class']
        
        if filename not in images:
            images[filename] = {'path': filepath, 'bboxes': [], 'labels': []}
        
        images[filename]['bboxes'].append(bbox)
        images[filename]['labels'].append(label)
    
    return list(images.values())

def load_dataset(image_folder):
    """Load images and corresponding annotations."""
    csv_path = os.path.join(image_folder, "_annotations.csv")
    df = load_annotations(csv_path)
    return parse_annotations(df, image_folder)

# Load train, validation, and test datasets
train_data = load_dataset(TRAIN_PATH)
valid_data = load_dataset(VALID_PATH)
test_data = load_dataset(TEST_PATH)

# Convert class names into integer labels
class_names = list(set(label for row in train_data for label in row['labels']))
#class_names = list(set(row['class'] for row in train_data))
class_to_index = {name: i for i, name in enumerate(class_names)}

def preprocess_data(item):
    """Load and preprocess an image and its bounding boxes."""
    img = cv2.imread(item['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (640, 480))  # Resize images to a fixed size
    img = img / 255.0  # Normalize pixel values
    
    # Convert labels to numeric indices
    bboxes = np.array(item['bboxes'], dtype=np.float32) / [640, 480, 640, 480]
    labels = np.array([class_to_index[label] for label in item['labels']], dtype=np.int32)
    
    return img, bboxes, labels

# Convert datasets
train_images, train_bboxes, train_labels = zip(*[preprocess_data(item) for item in train_data])
valid_images, valid_bboxes, valid_labels = zip(*[preprocess_data(item) for item in valid_data])

# Convert to TensorFlow datasets

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_bboxes, train_labels)).batch(32)
valid_ds = tf.data.Dataset.from_tensor_slices((valid_images, valid_bboxes, valid_labels)).batch(32)
