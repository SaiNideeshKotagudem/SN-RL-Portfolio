import numpy as np
import tensorflow as tf
from tf_agents.networks import actor_distribution_network, value_network

class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        weights = (1.0 / len(self.buffer) / probabilities[indices]) ** self.beta
        weights /= weights.max()  # Normalize for stability
        return experiences, weights.astype(np.float32), indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class EGACAgent:
    def __init__(self, observation_spec, action_spec, learning_rate=1e-3, gamma=0.99, clip_norm=0.5):
        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec, action_spec, activation_fn=tf.keras.activations.relu
        )
        self.value_net = value_network.ValueNetwork(observation_spec, activation_fn=tf.keras.activations.relu)
        self.value_net_target = value_network.ValueNetwork(observation_spec, activation_fn=tf.keras.activations.relu)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.gamma = gamma
        self.clip_norm = clip_norm
        self.experience_buffer = PrioritizedExperienceReplayBuffer(50000)

    def train(self, batch_size=64):
        if len(self.experience_buffer.buffer) < batch_size:
            return None, None

        experiences, weights, indices = self.experience_buffer.sample(batch_size)
        obs, actions, rewards, next_obs = zip(*experiences)
        obs = np.array(obs).astype(np.float32)
        actions = np.array(actions).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        next_obs = np.array(next_obs).astype(np.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # Train Critic (Value Network)
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.value_net(obs, training=True), axis=-1)
            next_values = tf.squeeze(self.value_net_target(next_obs, training=False), axis=-1)
            value_targets = tf.stop_gradient(rewards + self.gamma * next_values)
            value_loss = tf.reduce_mean(weights * tf.square(values - value_targets))

        grads = tape.gradient(value_loss, self.value_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.critic_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))

        # Train Actor
        with tf.GradientTape() as tape:
            action_dist = self.actor_net(obs, training=True)[0]
            sampled_actions = action_dist.sample()
            log_probs = action_dist.log_prob(sampled_actions)
            values = tf.squeeze(self.value_net(obs, training=False), axis=-1)
            advantages = tf.stop_gradient(value_targets - values)
            actor_loss = -tf.reduce_mean(weights * log_probs * advantages)

        actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.clip_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_net.trainable_variables))

        # Update priorities in buffer
        td_errors = np.abs(values.numpy() - value_targets.numpy()) + 1e-6
        self.experience_buffer.update_priorities(indices, td_errors.tolist())

        return value_loss.numpy(), actor_loss.numpy()

    def update_target_network(self, tau=0.95):
        for target_var, var in zip(self.value_net_target.trainable_variables, self.value_net.trainable_variables):
            target_var.assign(tau * target_var + (1 - tau) * var)

    def evaluate(self, env, num_episodes=5, render=False):
        total_rewards = []
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_reward = 0
            while not time_step.is_last():
                action_dist = self.actor_net(time_step.observation)[0]
                action = action_dist.mode()
                time_step = env.step(action.numpy())  # Ensure NumPy for compatibility
                episode_reward += time_step.reward.numpy()
                if render:
                    env.render()
            total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Evaluation -> Avg Reward: {avg_reward:.2f}, Std Dev: {std_reward:.2f}")
        return avg_reward, std_reward
        
