import numpy as np
import cv2


def group_by(fun: callable, items: list) -> dict:
    groups = {}

    for item in items:
        key = fun(item)

        if key not in groups:
            groups[key] = []

        groups[key].append(item)

    return groups


def convert_image_to_bgr(image: np.ndarray) -> np.ndarray:
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image
