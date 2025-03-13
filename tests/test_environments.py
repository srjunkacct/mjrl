import numpy as np
import gymnasium as gym
import pytest
from mjrl.utils.gym_env import GymEnv

# List of all environments to test
ENV_IDS = [
    'mjrl_point_mass-v0',
    'mjrl_swimmer-v0',
    'mjrl_reacher_sawyer-v0',
    'mjrl_peg_insertion-v0'
]

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_creation(env_id):
    """Test that environments can be created"""
    env = GymEnv(env_id)
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_seeding(env_id):
    """Test environment seeding for reproducibility"""
    env1 = GymEnv(env_id)
    env2 = GymEnv(env_id)
    
    # Set same seed
    seed = 42
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    np.testing.assert_array_equal(obs1, obs2)
    
    # Test that same actions lead to same observations
    action = env1.action_space.sample()
    next_obs1, reward1, term1, trunc1, info1 = env1.step(action)
    next_obs2, reward2, term2, trunc2, info2 = env2.step(action)
    np.testing.assert_array_equal(next_obs1, next_obs2)
    assert reward1 == reward2

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_stepping(env_id):
    """Test environment stepping with random actions"""
    env = GymEnv(env_id)
    obs, _ = env.reset()  # Ensure reset is called first
    
    # Test multiple steps
    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Basic checks
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check observation bounds if they exist
        if env.observation_space.bounded_below.all() and env.observation_space.bounded_above.all():
            assert (next_obs >= env.observation_space.low).all()
            assert (next_obs <= env.observation_space.high).all()
        
        # Reset if episode is done
        if terminated or truncated:
            obs, _ = env.reset()

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_reset(env_id):
    """Test environment reset functionality"""
    env = GymEnv(env_id)
    
    # Test multiple resets
    for _ in range(3):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_full_episode(env_id):
    """Test running a full episode"""
    env = GymEnv(env_id)
    obs, _ = env.reset()
    
    total_steps = 0
    episode_reward = 0
    terminated = truncated = False
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        total_steps += 1
        
        # Ensure we don't get stuck in an infinite loop
        assert total_steps <= env.horizon, f"Episode exceeded maximum horizon of {env.horizon}"

    # Basic episode checks
    assert total_steps > 0
    assert isinstance(episode_reward, (int, float)) 