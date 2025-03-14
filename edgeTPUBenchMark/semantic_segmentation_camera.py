

"""
usage

python3 examples/semantic_segmentation_camera.py   --model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite --fps 25
```
"""

import argparse
import cv2
import numpy as np
import time
from PIL import Image
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, default="test_data\deeplabv3_mnv2_pascal_quant_edgetpu.tflite" , help='Path of the segmentation model.')
    parser.add_argument('--fps', type=int, default=10, help='Desired processing framerate (frames per second).')
    args = parser.parse_args()
    
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)
    
    cap = cv2.VideoCapture(0)  # Open the camera
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    frame_interval = 1.0 / args.fps  # Time per frame
    last_frame_time = time.time()
    frame_count = 0
    fps_display = 0
    last_fps_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed_time = current_time - last_frame_time
        if elapsed_time < frame_interval:
            continue  # Skip processing until the next frame interval
        last_frame_time = current_time
        
        frame_count += 1
        if current_time - last_fps_time >= 1.0:
            fps_display = frame_count
            frame_count = 0
            last_fps_time = current_time
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resized_img = img.resize((width, height), Image.LANCZOS)
        common.set_input(interpreter, resized_img)
        
        interpreter.invoke()
        result = segment.get_output(interpreter)
        if len(result.shape) == 3:
            result = np.argmax(result, axis=-1)
            
        mask = label_to_color_image(result).astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        overlay = cv2.addWeighted(frame, 0.6, mask, 0.4, 0)
        
        # Display FPS on the visualization
        cv2.putText(overlay, f'FPS: {fps_display}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Segmentatie', overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

