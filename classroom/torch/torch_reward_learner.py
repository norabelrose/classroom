from ..reward_learner import RewardLearner
from torch import nn


class TorchRewardLearner(RewardLearner):
    """
    A reward model that uses a PyTorch neural network to predict the reward.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the reward model.
        """
        super().__init__()
        self.model = model

    def predict_reward(self, state, action, next_state):
        """
        Predict the reward for the given state, action, and next state.
        """
        return self.model(state, action, next_state)