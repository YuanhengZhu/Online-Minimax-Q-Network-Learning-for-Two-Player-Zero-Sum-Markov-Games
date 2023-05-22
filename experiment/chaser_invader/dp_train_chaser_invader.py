from rl import Dp
from myenv import ChaserInvaderDiscreteEnv

chaser_invader_discrete_env = ChaserInvaderDiscreteEnv()
dp_agent = Dp(
    policy='tabular',
    env=chaser_invader_discrete_env,
    gamma=0.95,
    sample_size=None,
    vi_error=0.01,
    vi_iters=500,
    seed=100)

#dp_agent.prepare_samples()

dp_agent.learn(
    train_save_freq=10,
    train_save_log_name='dp_chaser_invader')