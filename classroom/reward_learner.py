from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from typing import SupportsFloat


class RewardLearner(ABC):
    """
    Abstract class for reward models.
    """
    @abstractmethod
    def predict_reward(self, state, action, next_state) -> SupportsFloat:
        """
        Predict the reward for the given state, action, and next state.
        """

    @abstractmethod
    def learning_step(self, state, action, next_state, reward) -> None:
        """
        Update the model based on the given state, action, next state, and reward.
        """