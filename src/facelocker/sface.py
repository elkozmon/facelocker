import cv2
import numpy as np
import sys


class SFace:
    def __init__(
        self,
        model_path: str,
        dis_type: int = 0,
        backend_id: int = 0,
        target_id: int = 0,
    ):
        # Check inputs
        assert dis_type in [
            cv2.FaceRecognizerSF_FR_COSINE,
            cv2.FaceRecognizerSF_FR_NORM_L2,
        ], "Invalid dis_type"

        # Initialize model
        self._dis_type = dis_type
        self._model_path = model_path
        self._backend_id = backend_id
        self._target_id = target_id
        self._model = cv2.FaceRecognizerSF.create(
            model=self._model_path,
            config="",
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

        # Set thresholds
        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128

    def set_backend(self, backend_id: int, target_id: int):
        self._backend_id = backend_id
        self._target_id = target_id

        self._model = cv2.FaceRecognizerSF.create(
            model=self._model_path,
            config="",
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

    def crop(self, image: np.ndarray, face_box: list) -> np.ndarray:
        if face_box is None:
            return image
        else:
            return self._model.alignCrop(image, face_box)

    def features(self, image: np.ndarray) -> np.ndarray:
        return self._model.feature(image)

    def match(self, feature1: np.ndarray, feature2: np.ndarray) -> bool:
        score = self._model.match(feature1, feature2, self._dis_type)

        match self._dis_type:
            case cv2.FaceRecognizerSF_FR_COSINE:
                return score >= self._threshold_cosine

            case cv2.FaceRecognizerSF_FR_NORM_L2:
                return score <= self._threshold_norml2

            case _:
                sys.exit("Unreachable code")

    def match_any(
        self, features1: list[np.ndarray], features2: list[np.ndarray]
    ) -> bool:
        for feature1 in features1:
            for feature2 in features2:
                if self.match(feature1, feature2):
                    return True

        return False
