"""
Wrapper around a gymnasium env that provides convenience functions
"""

import gymnasium as gym
import numpy as np


class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon, action_space=None, observation_space=None):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon
        self.action_space = action_space
        self.observation_space = observation_space


class GymEnv(object):
    def __init__(self, env_name, act_repeat=1, obs_mask=None, obs_delay=0):
        """
        Initialize the environment wrapper.
        """
        self.env = gym.make(env_name)
        self.env_id = env_name
        self.act_repeat = act_repeat
        self.obs_mask = obs_mask
        self.obs_delay = obs_delay
        self._has_reset = False  # Initialize reset state
        self.env_infos = []
        self.rewards_sum = 0.0
        
        # Cache spaces
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space
        
        # Get horizon
        self._horizon = self.env.spec.max_episode_steps if self.env.spec is not None else float('inf')
        
        # Create EnvSpec with spaces
        self._spec = EnvSpec(
            self.observation_dim,
            self.action_dim,
            self.horizon,
            action_space=self.action_space,
            observation_space=self.observation_space
        )

    @property
    def action_dim(self):
        return self._action_space.shape[0]

    @property
    def observation_dim(self):
        return self._observation_space.shape[0]

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    @property
    def spec(self):
        return self._spec

    def _process_observation(self, obs):
        """Helper to process observation into correct format"""
        if obs is None:
            return None
        return np.array(obs, dtype=np.float32).flatten()  # Ensure observation is a flat numpy array

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        """
        if seed is not None:
            self.env.reset(seed=seed)  # Set the seed first
        
        # Reset internal state
        self.env_infos = []
        self.rewards_sum = 0.0
        
        # Reset the environment and get the initial observation
        obs, info = self.env.reset(options=options)
        
        # Process the observation
        obs = self._process_observation(obs)
        
        # Set the reset flag on both the wrapper and underlying environment
        self._has_reset = True
        if hasattr(self.env, '_has_reset'):
            self.env._has_reset = True
        
        return obs, info

    def reset_model(self, seed=None):
        """
        Reset the model to initial state.
        """
        if hasattr(self.env, 'reset_model'):
            obs = self.env.reset_model()
            obs = self._process_observation(obs)
            self._has_reset = True
            if hasattr(self.env, '_has_reset'):
                self.env._has_reset = True
            return obs
        else:
            obs, _ = self.reset(seed=seed)
            return obs

    def step(self, a):
        """
        Step the environment with the given action.
        """
        if not self._has_reset:
            raise gym.error.ResetNeeded("Cannot call env.step() before calling env.reset()")
        
        if self.act_repeat == 1:
            a = np.clip(a, self.action_space.low, self.action_space.high)
            obs, reward, terminated, truncated, info = self.env.step(a)
            obs = self._process_observation(obs)
            self.env_infos.append(info)
            self.rewards_sum += reward
            return obs, reward, terminated, truncated, info
        else:
            reward = 0.0
            for _ in range(self.act_repeat):
                a = np.clip(a, self.action_space.low, self.action_space.high)
                obs, r, terminated, truncated, info = self.env.step(a)
                obs = self._process_observation(obs)
                reward += r
                self.env_infos.append(info)
                if terminated or truncated:
                    break
            self.rewards_sum += reward
            return obs, reward, terminated, truncated, info

    def render(self):
        try:
            self.env.env.mujoco_render_frames = True
            self.env.env.mj_render()
        except:
            self.env.render()

    def set_seed(self, seed=123):
        try:
            self.env.reset(seed=seed)
        except AttributeError:
            self.env.reset(seed=seed)

    def get_obs(self):
        try:
            obs = self.env.env.get_obs()
        except:
            obs = self.env.env._get_obs()
        obs = self._process_observation(obs)
        return self.obs_mask * obs

    def get_env_infos(self):
        try:
            return self.env.env.get_env_infos()
        except:
            return {}

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.env.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.env.set_env_state(state_dict)
        except:
            raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
        except:
            raise NotImplementedError

    # ===========================================

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        try:
            self.env.env.visualize_policy(policy, horizon, num_episodes, mode)
        except:
            for ep in range(num_episodes):
                o, _ = self.reset()  # Updated to handle new reset signature
                d = False
                t = 0
                score = 0.0
                while t < horizon and d is False:
                    a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                    o, r, terminated, truncated, info = self.step(a)
                    d = terminated or truncated
                    score = score + r
                    self.render()
                    t = t+1
                print("Episode score = %f" % score)

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_env_state=None,
                        terminate_at_done=True,
                        seed=123):

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        for ep in range(num_episodes):
            o, _ = self.reset()  # Updated to handle new reset signature
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs()
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, terminated, truncated, info = self.step(a)
                done = terminated or truncated
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]
