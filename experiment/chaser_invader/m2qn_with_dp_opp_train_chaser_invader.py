from rl import M2qnOpp
from myenv import ChaserInvaderBoxEnv, ChaserInvaderDiscreteEnv
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

def dp_bct_func(obs):
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
    return dp.bct(discrete_obs)


m2qn_dp_agent = M2qnOpp(
    policy='mlp',
    env=ChaserInvaderBoxEnv(),
    gamma=0.95,
    learning_rate=0.001,
    buffer_size=10000,
    exploration_fraction=10000 / 40000,
    exploration_final_eps=0.1,
    exploration_initial_eps=1.0,
    batch_size=16,
    double_q=True,
    learning_starts=1000,
    evaluated_network_update_freq=500,
    target_network_update_freq=100,
    bct_func=dp_bct_func,
    policy_kwargs={'layer_size': [50, 50]},
    seed=300)


m2qn_dp_agent.learn(
    total_timesteps=40000,
    train_save_freq=500,
    train_save_log_name='m2qn_with_dp_opp_chaser_invader',
    train_evaluate_freq=500,
    train_evaluate_func=lambda act_func: evaluate_chaser_invader(
        ChaserInvaderBoxEnv(), act_func, dp_bct_func, total_runs=1000),
    train_plot_func=plot_chaser_invader_evaluate_results
)