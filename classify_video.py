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
Performs continuous image classification with the camera video.

To classify things using a default MobileNet model, simply run the script:

    python3 classify_video.py

Or classify using your own model and labels file:

    python3 classify_video.py -m my_model.tflite

For information about the script options, run:

    python3 classify_video.py --help

For more instructions, see g.co/aiy/maker
"""

import argparse
from pycoral.utils.dataset import read_label_file
from aiymakerkit import vision
from aiymakerkit.utils import read_labels_from_metadata
import models


def main():
    classifier = vision.Classifier('models/my-model.tflite')
    labels = read_label_file('models/image_labels.txt')

    for frame in vision.get_frames():
        classes = classifier.get_classes(frame, top_k=1, threshold=0.3)
        if classes:
            score = classes[0].score
            label = labels.get(classes[0].id)
            vision.draw_label(frame, f'{label}: {round(score, 4)}')


if __name__ == '__main__':
    main()
