from rl import Dqn
from myenv import SoccerAEnv, SoccerEnv
from rl.common.rl_util import plot_soccer_evaluate_results, evaluate_soccer

from rl import Dp
dp = Dp(policy='tabular',
        env=SoccerEnv())
params, q_values = dp.load('.\\save\\train_dp_soccer_100')
dp.q_values = q_values
dp_bct_func = dp.bct

soccer_A_env_challenge_dp = SoccerAEnv(
    bct_func=dp_bct_func,
)

dqn_challenge_dp_agent = Dqn(
    policy='tabular',
    env=soccer_A_env_challenge_dp,
    gamma=0.95,
    learning_rate=0.1,
    buffer_size=10000,
    exploration_fraction=20000 / 600000,
    exploration_final_eps=0.1,
    exploration_initial_eps=1.0,
    batch_size=16,
    double_q=True,
    learning_starts=1000,
    target_network_update_freq=100,
    seed=None)



dqn_challenge_dp_agent.learn(
    total_timesteps=600000,
    train_save_freq=6000,
    train_save_log_name='dqn_challenge_dp_soccer',
    train_evaluate_freq=6000,
    train_evaluate_func=lambda act_func: evaluate_soccer(
        SoccerEnv(), act_func, dp_bct_func),
    train_plot_func=plot_soccer_evaluate_results
)