import cv2
from datetime import datetime
import os
from aiymakerkit import audio
import argparse

PICTURE_DIR = os.path.join(os.path.expanduser('~'), 'Pictures')
IMAGE_SIZE = (640, 480)

def capture_photo():
    cap = cv2.VideoCapture(0)  # Open default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
    
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now()
        filename = "VOICE_CAM_" + timestamp.strftime("%Y-%m-%d_%H%M%S") + '.png'
        filename = os.path.join(PICTURE_DIR, filename)
        cv2.imwrite(filename, frame)
        print('Saved', filename)
    else:
        print("Error: Couldn't capture image")
    
    cap.release()
    cv2.destroyAllWindows()

def handle_results(label, score):
    if label == 'start':
        return False
    elif label == 'foto':
        capture_photo()
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    args = parser.parse_args()
    
    try:
        audio.classify_audio(model_file=args.model_file, callback=handle_results)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()