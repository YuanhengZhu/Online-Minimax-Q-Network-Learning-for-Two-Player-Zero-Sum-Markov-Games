import numpy as np
import random
from rl.common.rl_util import minimax
import os, json

class Dp:
    """
    dynamic programming player
    """
    def __init__(self, policy, env, gamma=0.99, sample_size=None, vi_error=None, vi_iters=None,
                 learning_rate=5e-4, batch_size=32, dueling=True,
                 policy_kwargs=None, seed=None):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.sample_size = sample_size
        self.vi_error = vi_error
        self.vi_iters = vi_iters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dueling = dueling
        self.policy_kwargs = policy_kwargs
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed + 100)

        if self.policy == 'tabular':
            s_dim = self.env.observation_space.n
            a_dim = self.env.action_space.n
            b_dim = self.env.bction_space.n
            self.q_values = np.zeros([s_dim, a_dim, b_dim])
        elif policy == 'mlp':
            raise NotImplementedError

    def prepare_samples(self):
        if self.policy == 'tabular' and self.sample_size is None:
            self.sample_buffer = []
            for obs in range(self.env.observation_space.n):
                self.env.observation_to_state(obs)
                for act in range(self.env.action_space.n):
                    for bct in range(self.env.bction_space.n):
                        probs, obses_tp1, rews, dones = self.env.probability_model(act, bct)
                        self.sample_buffer.append((obs, act, bct, probs, obses_tp1, rews, dones))
        else:
            raise NotImplementedError

    def _train_step(self):
        if self.policy == 'tabular':
            update_q_values = np.zeros((self.env.observation_space.n,
                                        self.env.action_space.n,
                                        self.env.bction_space.n))
            for obs, act, bct, probs, obses_tp1, rews, dones in self.sample_buffer:
                for prob, obs_tp1, rew, done in zip(probs, obses_tp1, rews, dones):
                    if done:
                        update_q_values[obs, act, bct] += prob * rew
                    else:
                        prob_act_tp1 = minimax(self.q_values[obs_tp1, :, :])        # (nA,)
                        q_tp1_best = np.matmul(prob_act_tp1, self.q_values[obs_tp1, :, :])      # (nB,)
                        q_tp1_best_worst = np.amin(q_tp1_best)
                        update_q_values[obs, act, bct] += prob * (rew + self.gamma * q_tp1_best_worst)

            error = np.amax(np.absolute(self.q_values - update_q_values))
            self.q_values = update_q_values
            return error
        else:
            raise NotImplementedError


    def learn(self, train_save_freq=None, train_save_log_name='dp'):
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

        num_iter = 0
        error = 0
        for _ in range(self.vi_iters):
            error = self._train_step()
            num_iter += 1

            if num_iter % int(self.vi_iters / 20) == 0:
                print('DP uses vi for {} iterations'.format(num_iter))

            if can_save and num_iter % train_save_freq == 0:
                self.save(os.path.join(train_save_path, str(num_iter)))

            if error < self.vi_error:
                print('DP error {} reaches the lowest error bound {}'.format(
                    error, self.vi_error
                ))
                break

        if can_save:
            self.save(train_save_path)

        if error >= self.vi_error:
            print('DP can not reach the lowest error bound {} within {} iterations'.format(
                self.vi_error, self.vi_iters
            ))
            print('final error is {}'.format(error))
            return False
        else:
            return True


    def act(self, obs):
        if self.policy == 'tabular':
            Q = self.q_values[obs, :, :]
        elif self.policy == 'mlp':
            Q = self.q_values.predict(np.array(obs))[0]
        else:
            raise NotImplementedError
        prob = minimax(Q)
        return np.random.choice(range(self.env.action_space.n), p=prob)

    def bct(self, obs):
        if self.policy == 'tabular':
            Q = self.q_values[obs, :, :]
        elif self.policy == 'mlp':
            Q = self.q_values.predict(np.array(obs))[0]    # (nA,nB)
        else:
            raise NotImplementedError
        prob = minimax(-Q.T)
        return np.random.choice(range(self.env.bction_space.n), p=prob)


    def save(self, save_path):
        params = {
            "dueling": self.dueling,
            "sample_size": self.sample_size,
            "batch_size": self.batch_size,
            "vi_error": self.vi_error,
            "vi_iters": self.vi_iters,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "observation_space": self.env.observation_space.shape,
            "action_space": self.env.action_space.n,
            "bction_space": self.env.bction_space.n,
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
            q_values = np.load(os.path.join(load_path, "model.npy"), allow_pickle=True)
        elif params["policy"] == 'mlp':
            q_values = self._build_policy(params["observation_space"],
                                          params["action_space"],
                                          params["bction_space"],
                                          params["policy_kwargs"])
            q_values.load_weights(os.path.join(load_path, "model"))
        else:
            raise NotImplementedError

        return params, q_values


if __name__ == '__main__':
    from myenv import SoccerEnv

    dp = Dp(policy='tabular',
              env=SoccerEnv(),
              gamma=0.95,
              sample_size=None,
              vi_error=0.01,
              vi_iters=500,
              seed=100)
    params, q_values = dp.load('..\\..\\experiment\\soccer\\save\\train_dp_soccer_100')
    dp.q_values = q_values

    dp.env.seed()

    obs = dp.env.reset()
    dp.env.render()
    done = False
    while not done:
        act = dp.act(obs)
        bct = dp.bct(obs)
        new_obs, rew, done, info = dp.env.step(act, bct)

        obs = new_obs
        dp.env.render()


