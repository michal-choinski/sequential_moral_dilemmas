from collections import OrderedDict
import cv2
import logging
import numpy as np
import gym
from typing import Any, List

from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.models.preprocessors import Preprocessor, get_preprocessor
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)


class DictFlatteningPreprocessor(Preprocessor):
    """Preprocesses each dict value, then flattens it all into a vector.
    RLlib models will unpack the flattened output before _build_layers_v2().
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        assert isinstance(self._obs_space, gym.spaces.Dict)
        size = 0
        self.preprocessors = []
        for space in self._obs_space.spaces.values():
            logger.debug("Creating wrapped preprocessor for {}".format(space))
            preprocessor = get_preprocessor(space)(space, self._options)
            self.preprocessors.append(preprocessor)
            size += preprocessor.size
        return (size, )

    @override(Preprocessor)
    def transform(self, observation: TensorType) -> np.ndarray:
        self.check_shape(observation)
        array = np.zeros(self.shape, dtype=np.float32)
        self.write(observation, array, 0)
        return array

    @override(Preprocessor)
    def write(self, observation: TensorType, array: np.ndarray,
              offset: int) -> None:
        if not isinstance(observation, OrderedDict):
            observation = OrderedDict(sorted(observation.items()))
        assert len(observation) == len(self.preprocessors), \
            (len(observation), len(self.preprocessors))
        for o, p in zip(observation.values(), self.preprocessors):
            p.write(o, array, offset)
            offset += p.size