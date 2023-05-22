from rl import Dqn
from myenv import ChaserInvaderBoxEnv, ChaserInvaderDiscreteEnv, InvaderBoxEnv
from rl.common.rl_util import evaluate_chaser_invader, plot_chaser_invader_evaluate_results
import numpy as np


from rl import Dp
chaser_invader_discrete_env = ChaserInvaderDiscreteEnv()
dp = Dp(
    policy='tabular',
    env=chaser_invader_discrete_env
)
params, q_values = dp.load('.\\save\\train_dp_chaser_invader_100')
dp.q_values = q_values

def dp_chaser_func(obs):
    x_max = max(abs(chaser_invader_discrete_env.board_right),
                abs(chaser_invader_discrete_env.board_left))
    y_max = max(abs(chaser_invader_discrete_env.board_top),
                abs(chaser_invader_discrete_env.board_bottom))
    xA = int(x_max * obs[0][0])
    yA = int(y_max * obs[0][1])
    xB = int(x_max * obs[0][2])
    yB = int(y_max * obs[0][3])
    chaser_invader_discrete_env.reset(
        np.array([xA, yA]),
        np.array([xB, yB])
    )
    discrete_obs = chaser_invader_discrete_env.state_to_observation()
    return dp.act(discrete_obs)

invader_env = InvaderBoxEnv(
    chaser_func=dp_chaser_func,
)

dqn_challenge_dp_agent = Dqn(
    policy='mlp',
    env=invader_env,
    gamma=0.95,
    learning_rate=0.001,
    buffer_size=10000,
    exploration_fraction=10000 / 100000,
    exploration_final_eps=0.1,
    exploration_initial_eps=1.0,
    batch_size=16,
    double_q=True,
    learning_starts=1000,
    target_network_update_freq=100,
    policy_kwargs={'layer_size': [50, 50]},
    seed=None)



dqn_challenge_dp_agent.learn(
    total_timesteps=100000,
    train_save_freq=1000,
    train_save_log_name='dqn_challenge_dp_chaser_invader',
    train_evaluate_freq=1000,
    train_evaluate_func=lambda act_func: evaluate_chaser_invader(
        ChaserInvaderBoxEnv(), dp_chaser_func, act_func),
    train_plot_func=plot_chaser_invader_evaluate_results
)