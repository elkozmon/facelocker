#!/usr/bin/env python

import argparse
import cv2
import os
import logging
import signal
import sys
import time
import json

import utils
from feature_loader import FeatureLoader
from loginctl import LoginCtl
from counter import Counter
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
        prog="facelocker",
        description="Locks user sessions when unable to recognize users face on camera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--yunet", type=str, default=os.path.join("models", "face_detection_yunet_2023mar.onnx"), help="yunet model")
    parser.add_argument("--sface", type=str, default=os.path.join("models", "face_recognizer_fast.onnx"), help="sface model")
    parser.add_argument("--features", type=str, default=os.path.join("${HOME}", ".facelocker"), help="user feature files directory")
    parser.add_argument("--dry-run", action="store_true", help="do not lock screen")
    parser.add_argument("-t", "--threshold", type=int, default=5, help="number of consecutive frames without match to lock the screen")
    parser.add_argument("-i", "--interval", type=int, default=1000, help="interval between frame captures in milliseconds")
    parser.add_argument("-d", "--device", type=str, default="/dev/video0", help="video device")
    parser.add_argument("-b", "--backend", type=int, default=-1, help='''
            {:d}: OpenCV implementation + CPU,
            {:d}: CUDA + GPU (CUDA),
            {:d}: CUDA + GPU (CUDA FP16),
            {:d}: TIM-VX + NPU,
            {:d}: CANN + NPU
        '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument("-v", "--verbose", action="count", default=0)

    return parser

def get_video_capture(device: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    return cap

def main():
    # Parse arguments
    parser = new_parser()
    args = parser.parse_args()

    if args.threshold < 1:
        sys.exit("threshold must be greater than zero")

    # Configure logging
    handler = logging.StreamHandler(sys.stdout)

    logger = logging.getLogger("facelocker")
    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)
    logger.addHandler(handler)

    # Setup signals
    global stopped
    stopped = False

    def handler(signum, frame):
        logger.info(f"got signal {signum}, stopping")
        global stopped
        stopped = True
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

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

    # Initialize helpers
    feature_loader = FeatureLoader(args.features)
    frame_counter = Counter()
    loginctl = LoginCtl()
    sleep_time = args.interval / 1000

    # Start loop
    logger.info("starting facelocker")
    logger.debug(f"configuration: {json.dumps(vars(args), indent=4)}")

    while not stopped:
        # List unlocked sessions
        unlocked_sessions = loginctl.list_unlocked_sessions()

        # Remove sessions with users not present in feature set
        for session in unlocked_sessions:
            user_features = feature_loader.load(session.user)

            if len(user_features) == 0:
                logger.warning(f"ignoring session \"{session.id}\" of user \"{session.user}\" (reason: user features missing)")
                unlocked_sessions.remove(session)

        # Skip frame if no unlocked sessions
        if len(unlocked_sessions) == 0:
            logger.debug("resetting counter for all users (reason: there are no unlocked sessions)")
            frame_counter = Counter()
            time.sleep(sleep_time)
            continue

        # Capture image
        cap = get_video_capture(args.device)
        ret, image = cap.read()
        cap.release()

        if not ret:
            logger.error("unable to capture an image")
            time.sleep(sleep_time)
            continue

        # Convert image to BGR
        image = utils.convert_image_to_bgr(image)

        # Detect faces in the image
        h, w, _ = image.shape
        detection_model.set_input_size([w, h])
        cap_faces = detection_model.infer(image)
        cap_feats = [recognizer_model.features(recognizer_model.crop(image, face)) for face in cap_faces]

        # Handle users with unlocked sessions
        unlocked_sessions_by_user = utils.group_by(lambda x: x.user, unlocked_sessions)

        for user, user_unlocked_sessions in unlocked_sessions_by_user.items():
            logger.debug(f"user \"{user}\" last matched: {frame_counter.get(user)}")
            logger.debug(f"user \"{user}\" unlocked sessions: {len(user_unlocked_sessions)}")

            user_feats = feature_loader.load(user)
            user_match = recognizer_model.match_any(cap_feats, user_feats)

            if user_match:
                logger.debug(f"resetting counter for user \"{user}\" (reason: face recognized)")
                frame_counter.reset(user)
                continue

            logger.debug(f"incrementing counter for user \"{user}\" (reason: no face recognized)")
            user_frames = frame_counter.increment_and_get(user)

            # Lock sessions if threshold met
            if user_frames >= args.threshold:
                logger.debug(f"locking sessions for user \"{user}\" (reason: threshold met)")

                if not args.dry_run:
                    loginctl.lock_sessions(user_unlocked_sessions)

        # Sleep interval
        time.sleep(sleep_time)

if __name__ == '__main__':
    main()
