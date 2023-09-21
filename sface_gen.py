#!/usr/bin/env python

import os
import sys
import argparse
import cv2
import numpy as np

import utils
from yunet import YuNet
from sface import SFace

def new_parser():
    parser = argparse.ArgumentParser(
        prog="facelocker sface feature generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--yunet", type=str, default=os.path.join("models", "face_detection_yunet_2023mar.onnx"), help="yunet model")
    parser.add_argument("--sface", type=str, default=os.path.join("models", "face_recognizer_fast.onnx"), help="sface model")
    parser.add_argument("-i", "--image", help="face picture image path")
    parser.add_argument("-d", "--device", help="video device", default="/dev/video0")
    parser.add_argument("-o", "--output", help="output directory", default="features")

    return parser

def capture_image(device):
    # Get video capture
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # Capture image
    ret, image = cap.read()
    
    if not ret:
        return None

    return image

def main():
    parser = new_parser()
    args = parser.parse_args()

    # Get image
    if args.image:
        image = cv2.imread(args.image)
    else:
        image = capture_image(args.device)

    if image is None:
        sys.exit("unable to get an image")
    
    image = utils.convert_image_to_bgr(image)

    # Load models
    detection_model = YuNet(model_path=args.yunet)
    recognizer_model = SFace(model_path=args.sface)

    # Detect faces
    h, w, _ = image.shape
    detection_model.set_input_size([w, h])
    faces = detection_model.infer(image)

    if len(faces) == 0:
        sys.exit("unable to detect faces")

    # Extract features
    features = []
    for face in faces:
        face_image = recognizer_model.crop(image, face)
        face_features = recognizer_model.features(face_image)
        features.append([face_features, face_image])

    if len(features) == 0:
        sys.exit("unable to get face features")

    # Save features
    for i, feature in enumerate(features):
        basename = "face{:03}".format(i + 1)
        cv2.imwrite(os.path.join(args.output, f"{basename}.jpg"), feature[1])
        np.save(os.path.join(args.output, basename), feature[0])

if __name__ == '__main__':
    main()
