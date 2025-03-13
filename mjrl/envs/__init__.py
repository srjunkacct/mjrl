from gymnasium.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    'mjrl_point_mass-v0',
    entry_point='mjrl.envs.point_mass:PointMassEnv',
    max_episode_steps=50,
)

register(
    'mjrl_swimmer-v0',
    entry_point='mjrl.envs.swimmer:SwimmerEnv',
    max_episode_steps=500,
)

register(
    'mjrl_reacher_sawyer-v0',
    entry_point='mjrl.envs.reacher_sawyer:Reacher7DOFEnv',
    max_episode_steps=50,
)

register(
    'mjrl_peg_insertion-v0',
    entry_point='mjrl.envs.peg_insertion_sawyer:PegInsertionSawyerEnv',
    max_episode_steps=75,
)

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjrl.envs.point_mass import PointMassEnv
from mjrl.envs.swimmer import SwimmerEnv
from mjrl.envs.reacher_sawyer import Reacher7DOFEnv
from mjrl.envs.peg_insertion_sawyer import PegInsertionSawyerEnv
