# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performs live speech recognition with a microphone.

To run the script, pass a speech model as the only argument:

    python3 classify_audio.py soundclassifier_with_metadata.tflite

Specifically, the model must be based on BrowserFFT, which you can train
yourself at https://teachablemachine.withgoogle.com/train/audio

For more instructions, see g.co/aiy/maker
"""

from picamera import PiCamera
from datetime import datetime
import os


from aiymakerkit import audio
import argparse

PICTURE_DIR = os.path.join(os.path.expanduser('~'), 'Pictures')
IMAGE_SIZE = (640, 480)

#camera = picamera.PiCamera(resolution=IMAGE_SIZE)
camera = picamera(resolution=IMAGE_SIZE)



def capture_photo():
    timestamp = datetime.now()
    filename = "VOICE_CAM_" + timestamp.strftime("%Y-%m-%d_%H%M%S") + '.png'
    filename = os.path.join(PICTURE_DIR, filename)
    camera.capture(filename)
    print('Saved', filename)


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
        audio.classify_audio(model_file=args.model, callback=handle_results)
    finally:
        camera.close()

if __name__ == '__main__':
    main()

