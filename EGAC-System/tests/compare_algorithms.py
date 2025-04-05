import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network, value_network, q_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.utils import common

from egac import EGACAgent  # assuming lowercase file name 'egac.py'
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.dqn import dqn_agent  # placeholder for RAC


def create_env(env_name='LunarLander-v2'):
    env = suite_gym.load(env_name)
    return tf_py_environment.TFPyEnvironment(env)


def train_agent(agent_type, num_iterations=5000, collect_episodes_per_iter=5):
    env = create_env()
    eval_env = create_env()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    time_step_spec = env.time_step_spec()

    if agent_type == 'EGAC':
        agent = EGACAgent(observation_spec, action_spec)

    elif agent_type == 'PPO':
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec, action_spec)
        value_net = value_network.ValueNetwork(observation_spec)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        agent = ppo_clip_agent.PPOClipAgent(
            time_step_spec,
            action_spec,
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            normalize_rewards=True,
            train_step_counter=tf.Variable(0))
        agent.initialize()

    elif agent_type == 'SAC':
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec, action_spec)
        critic_net = value_network.ValueNetwork(observation_spec)
        agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(1e-3),
            critic_optimizer=tf.keras.optimizers.Adam(1e-3),
            alpha_optimizer=tf.keras.optimizers.Adam(1e-3),
            train_step_counter=tf.Variable(0))
        agent.initialize()

    elif agent_type == 'RAC':
        q_net = q_network.QNetwork(observation_spec, action_spec)
        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_network=q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0))
        agent.initialize()

    else:
        raise ValueError("Unsupported agent type")

    rewards = []
    for iteration in range(num_iterations):
        episode_rewards = []

        for _ in range(collect_episodes_per_iter):
            time_step = env.reset()
            total_reward = 0
            while not time_step.is_last():
                if agent_type == 'EGAC':
                    action_dist = agent.actor_net(time_step.observation)[0]
                    action = action_dist.sample()
                    next_time_step = env.step(action.numpy())
                    experience = (
                        time_step.observation.numpy(),
                        action.numpy(),
                        time_step.reward.numpy(),
                        next_time_step.observation.numpy()
                    )
                    agent.experience_buffer.push(experience)
                    time_step = next_time_step
                else:
                    action_step = agent.policy.action(time_step)
                    action = action_step.action
                    time_step = env.step(action)
                total_reward += time_step.reward.numpy()
            episode_rewards.append(total_reward)

        rewards.append(np.mean(episode_rewards))

        if agent_type == 'EGAC':
            agent.train()
            agent.update_target_network()

        if iteration % 100 == 0:
            print(f"[{agent_type}] Iteration {iteration}: Avg Reward = {np.mean(episode_rewards):.2f}")

    return rewards


# Train and evaluate
algorithms = ['EGAC', 'PPO', 'SAC', 'RAC']
results = {}

for algo in algorithms:
    print(f"\nTraining {algo}...")
    rewards = train_agent(algo, num_iterations=1000)
    results[algo] = rewards

# Plot results
plt.figure(figsize=(12, 8))
for algo in algorithms:
    plt.plot(results[algo], label=algo)
plt.xlabel("Iterations")
plt.ylabel("Average Episode Reward")
plt.title("Comparison of RL Algorithms on LunarLander-v2")
plt.legend()
plt.grid(True)
plt.show()
        
