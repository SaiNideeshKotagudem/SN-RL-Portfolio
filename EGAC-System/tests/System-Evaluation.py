import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_gym
from EGAC import EGACAgent, actor_distribution_network, value_network

# Environment
env_name = 'LunarLander-v2'
env = suite_gym.load(env_name)
observation_spec = env.observation_spec()
action_spec = env.action_spec()

# Load networks (match architecture used in training)
actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec, action_spec, activation_fn=tf.keras.activations.relu
)
value_net = value_network.ValueNetwork(observation_spec, activation_fn=tf.keras.activations.relu)
value_net_target = value_network.ValueNetwork(observation_spec, activation_fn=tf.keras.activations.relu)

# Load weights (update file names with the latest checkpoint)
actor_net.load_weights("model_checkpoint_20000.h5")
value_net.load_weights("value_net_20000.h5")

# Create agent
agent = EGACAgent(actor_net, value_net, value_net_target)

# Evaluation function
def evaluate_agent(agent, env, num_episodes=20):
    total_rewards = []

    for ep in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0

        while not time_step.is_last():
            action_dist = agent.actor_net(time_step.observation)[0]
            action = action_dist.mode()  # Use mode for evaluation (greedy)
            time_step = env.step(action)
            episode_reward += time_step.reward.numpy()
        
        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average Reward = {avg_reward:.2f}, Std = {std_reward:.2f}")

evaluate_agent(agent, env)
