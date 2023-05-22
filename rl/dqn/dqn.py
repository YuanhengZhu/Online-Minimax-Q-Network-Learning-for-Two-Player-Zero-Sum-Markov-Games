import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Add
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os
import json
from rl.common.rl_util import get_linear_epsilon_value


class Dqn:
    """
    :param policy: "tabular" or "mlp"
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer or tabular tuning parameter
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param exploration_initial_eps: (float) initial value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param dueling: (bool) whether to enable dueling network structure
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param evaluated_network_update_freq: (int) update the evaluated network every `evaluated_network_update_freq` steps
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param policy_kwargs: (dict) structure information for mlp policy, e.g. (list int) a list of layer sizes
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed.
    """
    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, batch_size=32, double_q=True, dueling=True,
                 learning_starts=1000, target_network_update_freq=500,
                 policy_kwargs=None, seed=None):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.batch_size = batch_size
        self.double_q = double_q
        self.dueling = dueling
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.policy_kwargs = policy_kwargs
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed+100)

        a_dim = self.env.action_space.n
        if self.policy == 'tabular':
            s_dim = self.env.observation_space.n
            self.q_values = np.zeros([s_dim, a_dim])
            self.target_values = np.zeros([s_dim, a_dim])
            if self.double_q:
                self.double_values = self.q_values
        elif policy == 'mlp':
            s_dim = self.env.observation_space.shape
            self.q_values = self._build_policy(
                s_dim, a_dim, self.policy_kwargs['layer_size'])
            self.target_values = self._build_policy(
                s_dim, a_dim, self.policy_kwargs['layer_size'])
            if self.double_q:
                self.double_values = self.q_values

        self.replay_buffer = deque(maxlen=self.buffer_size)


    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_policy(self, input_shape, num_actionsA, layer_size):
        inputs = Input(shape=input_shape)
        action_out = Dense(layer_size[0], activation='relu')(inputs)
        for i in range(1, len(layer_size)):
            action_out = Dense(layer_size[i], activation='relu')(action_out)
        action_scores = Dense(num_actionsA)(action_out)

        if self.dueling:
            state_out = Dense(layer_size[0], activation='relu')(inputs)
            for i in range(1, len(layer_size)):
                state_out = Dense(layer_size[i], activation='relu')(state_out)
            state_score = Dense(1)(state_out)
            action_scores_centered = Lambda(lambda action_scores: action_scores - tf.reduce_mean(
                action_scores, axis=-1, keep_dims=True))(action_scores)
            state_score = Lambda(lambda state_score: tf.tile(
                state_score, [1, num_actionsA]))(state_score)
            q_out = Add()([state_score, action_scores_centered])
        else:
            q_out = action_scores

        model = Model(inputs=inputs, outputs=q_out)
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def replay_buffer_add(self, obs, act, rew, n_obs, done):
        self.replay_buffer.append((obs, act, rew, n_obs, done))


    def replay_buffer_sample(self, batch_size):
        samples = random.sample(self.replay_buffer, batch_size)
        obses, acts, rews, obses_tp1, dones = [], [], [], [], []
        for sample in samples:
            obses.append(np.squeeze(sample[0]))
            acts.append(sample[1])
            rews.append(sample[2])
            obses_tp1.append(np.squeeze(sample[3]))
            dones.append(sample[4])
        return (np.array(obses), np.array(acts), np.array(rews),
                np.array(obses_tp1), np.array(dones))


    def _train_step(self, obses_t, actions, rewards, obses_tp1, dones):
        if self.policy == 'tabular':
            q_t_selected = self.q_values[obses_t,actions]
            q_tp1_target = self.target_values[obses_tp1,:]            # (batch, nA)
            if self.double_q:
                q_tp1_double = self.double_values[obses_tp1,:]        # (batch, nA)
                act_tp1 = np.argmax(q_tp1_double, axis=1)       # (batch,)
                act_tp1_onehot = np.zeros((self.batch_size, self.env.action_space.n))     # (batch, nA)
                act_tp1_onehot[range(self.batch_size), act_tp1] = 1     # (batch, nA)
                q_tp1_best = np.sum(q_tp1_target * act_tp1_onehot, axis=1)  # (batch,)
            else:
                q_tp1_best = np.amax(q_tp1_target, axis=1)      # (batch,)

            q_tp1_best = (1.0 - dones) * q_tp1_best

            q_t_selected_target = rewards + self.gamma * q_tp1_best
            td_error = q_t_selected_target - q_t_selected

            self.q_values[obses_t,actions] += self.learning_rate * td_error
            return np.sum(td_error), None

        elif self.policy == 'mlp':
            q_t = self.q_values.predict(obses_t)       # (batch, nA)
            q_tp1_target = self.target_values.predict(obses_tp1)    # (batch, nA)
            if self.double_q:
                q_tp1_double = self.double_values.predict(obses_tp1)        # (batch, nA, nB)
                act_tp1 = np.argmax(q_tp1_double, axis=1)  # (batch,)
                act_tp1_onehot = np.zeros((self.batch_size, self.env.action_space.n))  # (batch, nA)
                act_tp1_onehot[range(self.batch_size), act_tp1] = 1  # (batch, nA)
                q_tp1_best = np.sum(q_tp1_target * act_tp1_onehot, axis=1)  # (batch,)
            else:
                q_tp1_best = np.amax(q_tp1_target, axis=1)      # (batch,)

            q_tp1_best = (1.0 - dones) * q_tp1_best

            q_t_selected_target = rewards + self.gamma * q_tp1_best
            td_error = q_t[range(self.batch_size),actions] - q_t_selected_target

            q_t[range(self.batch_size), actions] = q_t_selected_target
            history = self.q_values.fit(obses_t, q_t, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            return np.sum(td_error), loss


    def act(self, obs, update_eps=0.0):
        if np.random.rand() < update_eps:
            return np.random.choice(range(self.env.action_space.n))
        else:
            if self.policy == 'tabular':
                Q = self.q_values[obs, :]
            elif self.policy == 'mlp':
                Q = self.q_values.predict(obs)[0]
            else:
                raise NotImplementedError
            a = np.argmax(Q)
            return a

    def learn(self, total_timesteps, train_save_freq=None, train_save_log_name='dqn',
              train_evaluate_freq=None, train_evaluate_func=None, train_plot_func=None):
        if train_save_freq is not None and train_save_log_name is not None:
            can_save = True
            if self.seed is None:
                train_save_path = os.path.join(
                    os.getcwd(), 'save', 'train_{}'.format(train_save_log_name))
            else:
                train_save_path = os.path.join(
                    os.getcwd(), 'save', 'train_{}_{}'.format(train_save_log_name, self.seed))
            os.makedirs(train_save_path, exist_ok=True)
        else:
            can_save = False

        if train_evaluate_freq is not None:
            can_evalaute = True
            train_evaluate_results = []
            train_evaluate_timesteps = []
        else:
            can_evalaute = False

        obs = self.env.reset()
        num_timesteps = 0

        for _ in range(total_timesteps):
            update_eps = get_linear_epsilon_value(
                num_timesteps,
                self.exploration_fraction * total_timesteps,
                self.exploration_initial_eps, self.exploration_final_eps)
            act = self.act(obs, update_eps)
            new_obs, rew, done, info = self.env.step(act)
            num_timesteps += 1

            self.replay_buffer_add(obs, act, rew, new_obs, done)

            if len(self.replay_buffer) > self.batch_size \
                and num_timesteps > self.learning_starts:
                experience = self.replay_buffer_sample(self.batch_size)
                obses_t, actions, rewards, obses_tp1, dones = experience
                self._train_step(obses_t, actions, rewards, obses_tp1, dones)

                if num_timesteps % self.target_network_update_freq == 0:
                    if self.policy == 'tabular':
                        self.target_values = np.copy(self.q_values)
                    elif self.policy == 'mlp':
                        self.target_values.set_weights(self.q_values.get_weights())

            if can_evalaute and num_timesteps % train_evaluate_freq == 0:
                res = train_evaluate_func(lambda obs: self.act(obs, 0.0))
                train_evaluate_results.append(res)
                train_evaluate_timesteps.append(num_timesteps)
                train_plot_func(train_evaluate_timesteps, train_evaluate_results)

            if can_save and num_timesteps % train_save_freq == 0:
                os.makedirs(os.path.join(train_save_path, str(num_timesteps)), exist_ok=True)
                self.save(os.path.join(train_save_path, str(num_timesteps)))

            if done:
                obs = self.env.reset()
            else:
                obs = new_obs

        if can_save:
            self.save(train_save_path)
            train_evaluate = {
                'timesteps': train_evaluate_timesteps,
                'evaluate_results': train_evaluate_results
            }
            with open(os.path.join(train_save_path, "train_evaluate_results"), 'w') as f:
                json.dump(train_evaluate, f)


    def save(self, save_path):
        params = {
            "double_q": self.double_q,
            "dueling": self.dueling,
            "learning_starts": self.learning_starts,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "observation_space": self.env.observation_space.shape,
            "action_space": self.env.action_space.n,
            "policy": self.policy,
            "seed": self.seed,
            "policy_kwargs": self.policy_kwargs
        }
        with open(os.path.join(save_path, "params"), 'w') as f:
            json.dump(params, f)

        if self.policy == 'tabular':
            np.save(os.path.join(save_path, "model"), self.q_values)
        elif self.policy == 'mlp':
            self.q_values.save_weights(os.path.join(save_path, "model"))


    def load(self, load_path):
        f = open(os.path.join(load_path, "params"))
        params = json.load(f)

        if params["policy"] == 'tabular':
            q_values = np.load(os.path.join(load_path, "model"), allow_pickle=True)
        elif params["policy"] == 'mlp':
            q_values = self._build_policy(params["observation_space"],
                                          params["action_space"],
                                          params["policy_kwargs"])
            q_values.load_weights(os.path.join(load_path, "model"))

        return params, q_values


if __name__ == '__main__':

    def TestUpdatePolicy():
        m = M2qnPlayer(1, 2, 2)
        m.Qtab[0] = [[0, 1], [1, 0.5]]
        distr = m.PolicyA(0,m.Qtab)
        print(distr)

    TestUpdatePolicy()
