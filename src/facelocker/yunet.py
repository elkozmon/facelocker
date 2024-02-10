import cv2
from typing import List, Tuple


class YuNet:
    def __init__(
        self,
        model_path: str,
        input_size: List[int] = [320, 320],
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = 0,
        target_id: int = 0,
    ):
        self._model_path = model_path
        self._input_size = tuple(input_size)
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id

        # Initialize model
        self._model = cv2.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def set_backend(self, backend_id: int, target_id: int):
        self._backend_id = backend_id
        self._target_id = target_id

        self._model = cv2.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

    def set_input_size(self, input_size: List[int]):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image) -> List[Tuple]:
        _, results = self._model.detect(image)

        results = results if results is not None else []

        return results
