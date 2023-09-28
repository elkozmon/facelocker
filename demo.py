#!/usr/bin/env python

import argparse
import cv2
import os
import glob
import sys
import numpy as np

import utils
from yunet import YuNet
from sface import SFace

backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

def new_parser():
    parser = argparse.ArgumentParser(
        prog="facelocker demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--yunet", type=str, default=os.path.join("models", "face_detection_yunet_2023mar.onnx"), help="yunet model")
    parser.add_argument("--sface", type=str, default=os.path.join("models", "face_recognizer_fast.onnx"), help="sface model")
    parser.add_argument("--features", type=str, default="features", help="directory with face feature files")
    parser.add_argument("-d", "--device", type=str, default="/dev/video0", help="video device")
    parser.add_argument("-b", "--backend", type=int, default=0, help='''
            {:d}: OpenCV implementation + CPU,
            {:d}: CUDA + GPU (CUDA),
            {:d}: CUDA + GPU (CUDA FP16),
            {:d}: TIM-VX + NPU,
            {:d}: CANN + NPU
        '''.format(*[x for x in range(len(backend_target_pairs))]))

    return parser

def match_face(model: SFace, features: dict, image: np.ndarray, face: np.ndarray) -> str:
    face_image = model.crop(image, face)
    face_feature = model.features(face_image)

    # Find match in dictionary
    for id, feature in features.items():
        if model.match(face_feature, feature):
            return id

    return None

def load_features(dir: str) -> dict:
    features = {}

    for file in glob.glob(os.path.join(dir, "*.npy")):
        try:
            feature = np.load(file)
            id = os.path.splitext(os.path.basename(file))[0]
            features[id] = feature
        except Exception as e:
            sys.exit(f"failed to load {file}: {e}")

    return features

def main():
    parser = new_parser()
    args = parser.parse_args()

    # Get video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # Setup models
    backend_id = backend_target_pairs[args.backend][0]
    target_id = backend_target_pairs[args.backend][1]

    detection_model = YuNet(
        model_path=args.yunet,
        input_size=[0, 0],
        backend_id=backend_id,
        target_id=target_id)

    recognizer_model = SFace(
        model_path=args.sface,
        backend_id=backend_id,
        target_id=target_id)

    # Prepare features dictionary
    features = load_features(args.features)
    
    while True:
        # Get image
        ret, image = cap.read()

        if not ret:
            sys.exit("unable to get image from video device")
        
        image = utils.convert_image_to_bgr(image)

        # Detect faces in the image
        h, w, _ = image.shape
        detection_model.set_input_size([w, h])
        faces = detection_model.infer(image)

        # Draw the rectangle around each face
        for face in faces:
            # Find match in dictionary
            match match_face(recognizer_model, features, image, face):
                case None:
                    label = "unknown"
                    color = (0, 0, 255)
                case id:
                    label = id
                    color = (0, 255, 0)

            # Draw bounding box of face
            box = list(map(int, face[:4]))
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # Draw the recognition results
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, label, position, font, scale, color, thickness, cv2.LINE_AA)

        # Show image
        cv2.imshow("facelocker demo", image)

        # Stop if escape key is pressed
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

if __name__ == '__main__':
    main()
