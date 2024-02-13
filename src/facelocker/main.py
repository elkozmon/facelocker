import os
import sys
import getpass
import typer
import logging
import signal
import cv2
import numpy as np
import time
import json
import systemd_watchdog

from typing import Optional
from typing_extensions import Annotated

from importlib.metadata import version
from importlib.resources import files

from . import utils
from .feature_loader import FeatureLoader
from .loginctl import LoginCtl
from .counter import Counter
from .yunet import YuNet
from .sface import SFace

app = typer.Typer()
state = {"yunet": None, "sface": None, "logger": None}

backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU],
]


def version_callback(value: bool):
    if value:
        typer.echo(version("facelocker"))
        raise typer.Exit()


@app.callback()
def main(
    yunet: Annotated[str, typer.Option(help="YuNet model file path")] = os.path.join(
        files("facelocker"),
        "models",
        "yunet",
        "face_detection_yunet_2023mar.onnx",
    ),
    sface: Annotated[str, typer.Option(help="SFace model file path")] = os.path.join(
        files("facelocker"),
        "models",
        "sface",
        "face_recognizer_fast.onnx",
    ),
    verbose: Annotated[Optional[int], typer.Option("--verbose", "-v", count=True)] = 0,
    version: Annotated[
        Optional[bool], typer.Option("--version", callback=version_callback)
    ] = None,
):
    """
    Locks systemd user session automatically when user isn't detected on camera
    """

    state["yunet"] = yunet
    state["sface"] = sface
    state["logger"] = get_logger(verbose)


@app.command()
def run(
    features: Annotated[
        str, typer.Option(help="Feature files directory")
    ] = os.path.join("${HOME}", ".facelocker"),
    threshold: Annotated[
        int,
        typer.Option(
            "--threshold",
            "-t",
            help="Number of consecutive frames without match to lock the screen",
        ),
    ] = 5,
    interval: Annotated[
        int,
        typer.Option(
            "-i", "--interval", help="Interval between frame captures in milliseconds"
        ),
    ] = 1000,
    device: Annotated[
        str, typer.Option("--device", "-d", help="Video capture device path")
    ] = "/dev/video0",
    backend: Annotated[
        int,
        typer.Option(
            help="""
            {:d}: OpenCV implementation + CPU,
            {:d}: CUDA + GPU (CUDA),
            {:d}: CUDA + GPU (CUDA FP16),
            {:d}: TIM-VX + NPU,
            {:d}: CANN + NPU
        """.format(
                *[x for x in range(len(backend_target_pairs))]
            )
        ),
    ] = 0,
    dry_run: Annotated[
        bool, typer.Option("--dry-run/", help="Don't lock sessions")
    ] = False,
):
    """
    Start facelocker
    """

    if threshold < 1:
        sys.exit("threshold must be greater than zero")
    
    # Setup watchdog
    wd = systemd_watchdog.watchdog()

    if not wd.is_enabled:
        state["logger"].info(f"watchdog not enabled")
    else:
        wd_usec = int(os.environ.get('WATCHDOG_USEC', 0)) / 1000
        if interval >= wd_usec:
            sys.exit("interval equal or longer than watchdog timeout")

    # Setup signals
    global stopped
    stopped = False

    def handler(signum, frame):
        state["logger"].info(f"got signal {signum}, stopping")
        global stopped
        stopped = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Setup models
    backend_id = backend_target_pairs[backend][0]
    target_id = backend_target_pairs[backend][1]

    detection_model = YuNet(
        model_path=state["yunet"],
        input_size=[0, 0],
        backend_id=backend_id,
        target_id=target_id,
    )

    recognizer_model = SFace(
        model_path=state["sface"], backend_id=backend_id, target_id=target_id
    )

    # Initialize helpers
    feature_loader = FeatureLoader(features)
    frame_counter = Counter()
    loginctl = LoginCtl()
    sleep_time = interval / 1000
    users_ignored = set()

    # Start loop
    state["logger"].info("starting facelocker")
    wd.status("starting facelocker")
    wd.ready()
    wd.notify()

    while not stopped:
        # List unlocked sessions
        unlocked_sessions = loginctl.list_unlocked_sessions(skip_users=users_ignored)

        # Remove sessions with users not present in feature set
        for session in unlocked_sessions:
            user_features = feature_loader.load(session.user)

            if len(user_features) == 0:
                state["logger"].warning(
                    f'ignoring sessions of user "{session.user}" (reason: user features missing)'
                )
                users_ignored.add(session.user)
                unlocked_sessions.remove(session)

        # Skip frame if no unlocked sessions
        if len(unlocked_sessions) == 0:
            state["logger"].debug(
                "resetting counter for all users (reason: there are no unlocked sessions)"
            )
            frame_counter = Counter()
            time.sleep(sleep_time)
            wd.notify()
            continue

        # Capture image
        cap = get_video_capture(device)
        ret, image = cap.read()
        cap.release()

        if not ret:
            state["logger"].error("unable to capture an image")
            time.sleep(sleep_time)
            continue

        # Convert image to BGR
        image = utils.convert_image_to_bgr(image)

        # Detect faces in the image
        h, w, _ = image.shape
        detection_model.set_input_size([w, h])
        cap_faces = detection_model.infer(image)
        cap_feats = [
            recognizer_model.features(recognizer_model.crop(image, face))
            for face in cap_faces
        ]

        # Handle users with unlocked sessions
        unlocked_sessions_by_user = utils.group_by(lambda x: x.user, unlocked_sessions)

        for user, user_unlocked_sessions in unlocked_sessions_by_user.items():
            state["logger"].debug(
                f'user "{user}" last matched: {frame_counter.get(user)}'
            )
            state["logger"].debug(
                f'user "{user}" unlocked sessions: {len(user_unlocked_sessions)}'
            )

            user_feats = feature_loader.load(user)
            user_match = recognizer_model.match_any(cap_feats, user_feats)

            if user_match:
                state["logger"].debug(
                    f'resetting counter for user "{user}" (reason: face recognized)'
                )
                frame_counter.reset(user)
                continue

            state["logger"].debug(
                f'incrementing counter for user "{user}" (reason: no face recognized)'
            )
            user_frames = frame_counter.increment_and_get(user)

            # Lock sessions if threshold met
            if user_frames >= threshold:
                state["logger"].debug(
                    f'locking sessions for user "{user}" (reason: threshold met)'
                )

                if not dry_run:
                    loginctl.lock_sessions(user_unlocked_sessions)

        # Notify watchdog
        wd.notify()

        # Sleep interval
        time.sleep(sleep_time)


@app.command()
def test(
    features: Annotated[
        str, typer.Option(help="Feature files directory")
    ] = os.path.join("${HOME}", ".facelocker"),
    user: Annotated[
        str, typer.Option("--user", "-u", help="User whose feature files to use")
    ] = getpass.getuser(),
    interval: Annotated[
        int,
        typer.Option(
            "-i", "--interval", help="Interval between frame captures in milliseconds"
        ),
    ] = 1000,
    device: Annotated[
        str, typer.Option("--device", "-d", help="Video capture device path")
    ] = "/dev/video0",
    backend: Annotated[
        int,
        typer.Option(
            help="""
            {:d}: OpenCV implementation + CPU,
            {:d}: CUDA + GPU (CUDA),
            {:d}: CUDA + GPU (CUDA FP16),
            {:d}: TIM-VX + NPU,
            {:d}: CANN + NPU
        """.format(
                *[x for x in range(len(backend_target_pairs))]
            )
        ),
    ] = 0,
):
    """
    Test face detection
    """

    # Setup signals
    global stopped
    stopped = False

    def handler(signum, frame):
        state["logger"].info(f"got signal {signum}, stopping")
        global stopped
        stopped = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Setup models
    backend_id = backend_target_pairs[backend][0]
    target_id = backend_target_pairs[backend][1]

    detection_model = YuNet(
        model_path=state["yunet"],
        input_size=[0, 0],
        backend_id=backend_id,
        target_id=target_id,
    )

    recognizer_model = SFace(
        model_path=state["sface"], backend_id=backend_id, target_id=target_id
    )

    # Initialize helpers
    feature_loader = FeatureLoader(features)
    cap = get_video_capture(device)
    sleep_time = int(interval / 1000)

    while not stopped:
        # Get image
        ret, image = cap.read()

        if not ret:
            state["logger"].error("unable to capture an image")
            time.sleep(sleep_time)
            continue

        image = utils.convert_image_to_bgr(image)

        # Detect faces in the image
        h, w, _ = image.shape
        detection_model.set_input_size([w, h])
        faces = detection_model.infer(image)

        # Draw the rectangle around each face
        for face in faces:
            label = "unknown"
            color = (0, 0, 255)

            face_image = recognizer_model.crop(image, face)
            face_feature = recognizer_model.features(face_image)

            # Find match in dictionary
            for feature in feature_loader.load(user):
                if recognizer_model.match(face_feature, feature):
                    label = user
                    color = (0, 255, 0)

            # Draw bounding box of face
            box = list(map(int, face[:4]))
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # Draw the recognition results
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(
                image, label, position, font, scale, color, thickness, cv2.LINE_AA
            )

        # Show image
        cv2.imshow("facelocker test", image)

        # Stop if escape key is pressed
        if cv2.waitKey(sleep_time) & 0xFF == ord("q"):
            stopped = True


@app.command()
def feature(
    device: Annotated[
        str, typer.Option("--device", "-d", help="Video capture device path")
    ] = "/dev/video0",
    image: Annotated[str, typer.Option("--image", "-i", help="Face picture")] = None,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory")
    ] = os.getcwd(),
):
    """
    Generate facial feature files by capturing image using a video device or from an image file
    """

    # Get image
    if image:
        image = cv2.imread(image)
    else:
        cap = get_video_capture(device)
        _, image = cap.read()
        cap.release()

    if image is None:
        sys.exit("unable to get an image")

    image = utils.convert_image_to_bgr(image)

    # Load models
    detection_model = YuNet(model_path=state["yunet"])
    recognizer_model = SFace(model_path=state["sface"])

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
        cv2.imwrite(os.path.join(output, f"{basename}.jpg"), feature[1])
        np.save(os.path.join(output, basename), feature[0])


def get_logger(verbosity):
    logger = logging.getLogger("facelocker")
    handler = logging.StreamHandler(sys.stdout)

    logger.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
    logger.addHandler(handler)

    return logger


def get_video_capture(device: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    return cap


if __name__ == "__main__":
    app()
