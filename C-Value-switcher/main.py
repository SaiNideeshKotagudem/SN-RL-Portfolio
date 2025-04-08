import tensorflow as tf
import numpy as np
import gym

# ===== Confidence Evaluator =====

class ConfidenceEvaluator:
    def __init__(self, sigma_max=1.0, adaptive=True):
        self.sigma_max = sigma_max
        self.adaptive = adaptive

    def policy_entropy_confidence(self, action_probs):
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8))
        max_entropy = tf.math.log(tf.cast(tf.shape(action_probs)[0], tf.float32))
        return 1.0 - entropy / max_entropy

    def q_ensemble_confidence(self, q_values_ensemble):
        var = tf.math.reduce_variance(q_values_ensemble)
        return 1.0 - var / self.sigma_max

    def mc_dropout_confidence(self, q_values_mc):
        var = tf.math.reduce_variance(q_values_mc)
        return 1.0 - var / self.sigma_max

    def adaptive_weights(self, uncertainties):
        uncertainties = tf.stack(uncertainties)
        uncertainties = uncertainties / (tf.reduce_max(uncertainties) + 1e-8)
        return tf.nn.softmax(1.0 / (uncertainties + 1e-8))

    def compute(self, c1, c2, c3):
        if self.adaptive:
            raw_uncertainty = [1.0 - c1, 1.0 - c2, 1.0 - c3]
            weights = self.adaptive_weights(raw_uncertainty)
            return weights[0]*c1 + weights[1]*c2 + weights[2]*c3
        else:
            return (c1 + c2 + c3) / 3.0

# ===== Dummy Environments =====

class SimEnv:
    def __init__(self):
        pass

    def step(self, action):
        return np.random.random((4,)), np.random.random(), False, {}

    def reset(self):
        return np.random.random((4,))

class RealEnv:
    def __init__(self):
        pass

    def step(self, action):
        return np.random.random((4,)), np.random.random(), False, {}

    def reset(self):
        return np.random.random((4,))

# ===== Dummy Agent Networks =====

class DummyPolicyNet(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])

    def call(self, state):
        return self.net(state)

class DummyQEnsemble:
    def __init__(self, n, action_dim):
        self.models = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
                tf.keras.layers.Dense(action_dim)
            ]) for _ in range(n)
        ]

    def predict(self, state):
        return [model(state) for model in self.models]

class DropoutQNet(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(4,))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(action_dim)

    def call(self, state, training=False):
        x = self.fc1(state)
        x = self.dropout(x, training=training)
        return self.out(x)

# ===== Agent Loop with Switching & Hysteresis =====

def agent_loop(sim_env, real_env, policy_net, q_ensemble, dropout_q_net, evaluator):
    CONFIDENCE_ENTER_REAL = 0.7
    CONFIDENCE_EXIT_REAL = 0.6
    in_real = False
    state = sim_env.reset()

    for step in range(10):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = policy_net(state_tensor)[0]
        action = tf.random.categorical(tf.math.log([action_probs]), 1)[0, 0].numpy()

        # Confidence components
        c1 = evaluator.policy_entropy_confidence(action_probs)
        q_vals = tf.stack([q[0, action] for q in q_ensemble.predict(state_tensor)])
        c2 = evaluator.q_ensemble_confidence(q_vals)
        mc_vals = tf.stack([dropout_q_net(state_tensor, training=True)[0, action] for _ in range(10)])
        c3 = evaluator.mc_dropout_confidence(mc_vals)

        conf = evaluator.compute(c1, c2, c3)

        # Hysteresis-based switching
        if in_real and conf < CONFIDENCE_EXIT_REAL:
            in_real = False
        elif not in_real and conf > CONFIDENCE_ENTER_REAL:
            in_real = True

        env = real_env if in_real else sim_env
        state, reward, done, _ = env.step(action)

        print(f"Step {step} | Action {action} | C1: {c1.numpy():.3f} | C2: {c2.numpy():.3f} | C3: {c3.numpy():.3f} | Conf: {conf.numpy():.3f} | Env: {'Real' if in_real else 'Sim'}")

        if done:
            state = sim_env.reset()

# ===== Run the Agent =====

action_dim = 4
sim_env = SimEnv()
real_env = RealEnv()
policy_net = DummyPolicyNet(action_dim)
q_ensemble = DummyQEnsemble(n=5, action_dim=action_dim)
dropout_q_net = DropoutQNet(action_dim)
evaluator = ConfidenceEvaluator(sigma_max=1.0, adaptive=True)

agent_loop(sim_env, real_env, policy_net, q_ensemble, dropout_q_net, evaluator)
