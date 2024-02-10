import numpy as np
import glob
import os
import sys
from string import Template


class FeatureLoader:
    def __init__(self, path):
        self._path_template = Template(path)
        self._user_features = {}

    def load(self, user: str) -> list[np.ndarray]:
        # Try cache
        if user in self._user_features:
            return self._user_features[user]

        # Initialize users list
        self._user_features[user] = []

        # Evaluate path template
        path = self._path_template.substitute(
            {"HOME": os.path.expanduser(f"~{user}"), "USER": user}
        )

        # Load features
        for file in glob.glob(os.path.join(path, "*.npy")):
            try:
                feature = np.load(file)
                self._user_features[user].append(feature)
            except Exception as e:
                sys.exit(f"failed to load {file}: {e}")

        return self._user_features[user]
