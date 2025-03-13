import argparse
import collections
import time
import cv2
import numpy as np
from PIL import Image
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])

def draw_object(frame, obj):
    """Tekent een detectiekader op het scherm."""
    x_min, y_min, x_max, y_max = obj.bbox
    label = f"{obj.label} ({obj.score:.2f})"
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def non_max_suppression(objects, iou_threshold=0.5):
    """Voert Non-Maximum Suppression uit om overlappende boxen te verwijderen."""
    if len(objects) == 0:
        return []
    
    # Sorteer objecten op score (hoog naar laag)
    objects = sorted(objects, key=lambda x: x.score, reverse=True)
    
    keep_objects = []
    while objects:
        best = objects.pop(0)
        keep_objects.append(best)
        
        new_objects = []
        for obj in objects:
            iou = compute_iou(best.bbox, obj.bbox)
            if iou < iou_threshold:  
                new_objects.append(obj)
        
        objects = new_objects
    
    return keep_objects

def compute_iou(bbox1, bbox2):
    """Bereken Intersection over Union (IoU) tussen twee bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path naar SSD model.')
    parser.add_argument('--label', help='Path naar labels file.')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Minimale score voor detectie.')
    args = parser.parse_args()
    
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.label) if args.label else {}
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kon camera niet openen.")
        return
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, scale = common.set_resized_input(interpreter, img.size, lambda size, img=img: img.resize(size, Image.NEAREST))
        interpreter.invoke()
        objs = detect.get_objects(interpreter, args.score_threshold, scale)
        
        # Converteer objecten naar een bruikbaar formaat
        detected_objects = [
            Object(labels.get(obj.id, ''), obj.score, [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax])
            for obj in objs
        ]
        
        # Pas Non-Maximum Suppression toe
        filtered_objects = non_max_suppression(detected_objects)
        
        # Teken alleen de gefilterde objecten
        for obj in filtered_objects:
            draw_object(frame, obj)
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
