import cv2
from datetime import datetime
import os
from aiymakerkit import audio
import argparse

PICTURE_DIR = 'Pictures'
IMAGE_SIZE = (1280, 960)

def capture_photo():
    print ("take picture")
    cap = cv2.VideoCapture(0)  # Open default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
    
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now()
        filename = "VOICE_CAM_" + timestamp.strftime("%Y-%m-%d_%H%M%S") + '.png'
        filename = os.path.join(PICTURE_DIR, filename)
        print (filename)
        cv2.imwrite(filename, frame)
        print('Saved', filename)
    else:
        print("Error: Couldn't capture image")
    
    cap.release()
    cv2.destroyAllWindows()

def handle_results(label, score):
    if label == '2 start':
        return False
    elif (label == '1 foto' and score > 0.7) :
        capture_photo()
    return True

def main():
    
    try:
        audio.classify_audio("models/soundclassifier_with_metadata.tflite", callback=handle_results)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
