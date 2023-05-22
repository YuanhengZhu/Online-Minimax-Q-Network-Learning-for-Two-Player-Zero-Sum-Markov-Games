from rl import M2qn
from myenv import SoccerEnv, SoccerBoxEnv
from rl.common.rl_util import plot_soccer_evaluate_results, evaluate_soccer
import numpy as np

m2qn_agent = M2qn(
    policy='mlp',
    env=SoccerBoxEnv(),
    gamma=0.95,
    learning_rate=0.001,
    buffer_size=10000,
    exploration_fraction=10000 / 20000,
    exploration_final_eps=0.1,
    exploration_initial_eps=1.0,
    batch_size=16,
    double_q=True,
    learning_starts=1000,
    evaluated_network_update_freq=100, #500,
    target_network_update_freq=100,
    policy_kwargs={'layer_size': [50, 50]},
    seed=200)

from rl import Dp
soccer_discrete_env = SoccerEnv()
dp = Dp(policy='tabular',
        env=soccer_discrete_env)
params, q_values = dp.load('.\\save\\train_dp_soccer_100')
dp.q_values = q_values

def dp_bct_func(obs):
    xA = int(soccer_discrete_env.board_width * obs[0][0])
    yA = int(soccer_discrete_env.board_height * obs[0][1])
    xB = int(soccer_discrete_env.board_width * obs[0][2])
    yB = int(soccer_discrete_env.board_height * obs[0][3])
    ball_owner = bool(obs[0][4])
    soccer_discrete_env.reset(
        position_A=np.array([xA,yA]),
        position_B=np.array([xB,yB]),
        ball_owner=ball_owner
    )
    discrete_obs = soccer_discrete_env.state_to_observation()
    return dp.bct(discrete_obs)


m2qn_agent.learn(
    total_timesteps=20000,
    train_save_freq=500,
    train_save_log_name='m2qn_soccer_box',
    train_evaluate_freq=500,
    train_evaluate_func=lambda act_func: evaluate_soccer(
        SoccerBoxEnv(), act_func, dp_bct_func, total_runs=1000),
    train_plot_func=plot_soccer_evaluate_results
    )