import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MeanCummulativeRewardCallback(BaseCallback):
    #     """
    #     Callback for saving a model (the check is done every ``check_freq`` steps)
    #     based on the training reward (in practice, we recommend using ``EvalCallback``).
    #     :param check_freq: (int)
    #     :param log_dir: (str) Path to the folder where the model will be saved.
    #       It must contains the file created by the ``Monitor`` wrapper.
    #     :param verbose: (int)
    #     """

    def __init__(self, check_freq, log_dir, verbose=False):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        print("log_dir: ", log_dir)
        super(MeanCummulativeRewardCallback, self).__init__(verbose)
        self.cumulative_rewards = []
        self.mean_cumulative_reward = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            print(
                "Num timesteps: {}".format(self.num_timesteps),
                "mean cumulative reward: ",
                self.mean_cumulative_reward,
            ) if self.verbose else None
            reward = float(self.locals["rewards"])
            self.cumulative_rewards.append(reward)
            self.mean_cumulative_reward = np.mean(self.cumulative_rewards)

        return True
