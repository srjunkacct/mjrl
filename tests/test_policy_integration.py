import numpy as np
import pytest
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils import tensor_utils

# List of all environments to test
ENV_IDS = [
    'mjrl_point_mass-v0',
    'mjrl_swimmer-v0',
    'mjrl_reacher_sawyer-v0',
    'mjrl_peg_insertion-v0'
]

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_policy_creation(env_id):
    """Test policy creation for each environment"""
    env = GymEnv(env_id)
    policy = MLP(env.spec, hidden_sizes=(32, 32), seed=123)
    
    # Basic policy checks
    assert policy.obs_dim == env.observation_dim
    assert policy.act_dim == env.action_dim
    assert len(policy.hidden_sizes) == 2
    assert policy.hidden_sizes[0] == 32

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_policy_action(env_id):
    """Test policy action generation"""
    env = GymEnv(env_id)
    policy = MLP(env.spec, hidden_sizes=(32, 32), seed=123)
    
    obs, _ = env.reset()
    action, info = policy.get_action(obs)
    
    # Check action properties
    assert isinstance(action, np.ndarray)
    assert action.shape == (env.action_dim,)
    assert isinstance(info, dict)
    assert 'evaluation' in info
    
    # Check if action is within bounds
    assert (action >= env.action_space.low).all()
    assert (action <= env.action_space.high).all()

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_policy_episode(env_id):
    """Test running full episodes with policy"""
    env = GymEnv(env_id)
    policy = MLP(env.spec, hidden_sizes=(32, 32), seed=123)
    
    # Run multiple episodes
    n_episodes = 3
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_length = 0
        
        while not done:
            action, _ = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1
            
            assert episode_length <= env.horizon

def test_policy_batch_processing():
    """Test policy with batch processing of observations"""
    env_id = ENV_IDS[0]  # Use point mass as example
    env = GymEnv(env_id)
    policy = MLP(env.spec, hidden_sizes=(32, 32), seed=123)
    
    # Create batch of observations
    batch_size = 10
    obs_batch = np.stack([env.reset()[0] for _ in range(batch_size)])
    
    # Get actions for batch
    actions, infos = policy.get_action_batch(obs_batch)
    
    # Check batch processing results
    assert actions.shape == (batch_size, env.action_dim)
    assert len(infos) == batch_size
    assert all('evaluation' in info for info in infos)

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_deterministic_policy(env_id):
    """Test deterministic policy behavior"""
    env = GymEnv(env_id)
    policy = MLP(env.spec, hidden_sizes=(32, 32), seed=123)
    
    obs, _ = env.reset(seed=42)
    
    # Get actions multiple times for same observation
    action1, _ = policy.get_action(obs)
    action2, _ = policy.get_action(obs)
    
    # Actions should be different (stochastic) but within bounds
    assert not np.array_equal(action1, action2)
    assert (action1 >= env.action_space.low).all() and (action1 <= env.action_space.high).all()
    assert (action2 >= env.action_space.low).all() and (action2 <= env.action_space.high).all()

@pytest.mark.parametrize("env_id", ENV_IDS)
def test_policy_evaluation_mode(env_id):
    """Test policy in evaluation mode"""
    env = GymEnv(env_id)
    policy = MLP(env.spec, hidden_sizes=(32, 32), seed=123)
    
    obs, _ = env.reset(seed=42)
    
    # Get evaluation actions
    _, info1 = policy.get_action(obs)
    _, info2 = policy.get_action(obs)
    
    # Evaluation actions should be deterministic
    assert np.array_equal(info1['evaluation'], info2['evaluation']) 