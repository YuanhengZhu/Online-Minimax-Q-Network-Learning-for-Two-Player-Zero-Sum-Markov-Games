from gym.envs.registration import register

from myenv.env_soccer import SoccerEnv
from myenv.env_soccer_box import SoccerBoxEnv
from myenv.env_soccer_A import SoccerAEnv

from myenv.env_chaser_invader_box import ChaserInvaderBoxEnv
from myenv.env_chaser_invader_discrete import ChaserInvaderDiscreteEnv
from myenv.env_invader_box import InvaderBoxEnv

register(
    id = "SoccerBoxEnv-v0",
    entry_point = "myenv.env_soccer_box:SoccerBoxEnv"
)

register(
    id = "SoccerEnv-v0",
    entry_point = "myenv.env_soccer:SoccerEnv"
)

register(
    id = "SoccerAEnv-v0",
    entry_point = "myenv.env_soccer_A:SoccerAEnv"
)

register(
    id = "ChaserInvaderBoxEnv-v0",
    entry_point = "myenv.env_chaser_invader_box:ChaserInvaderBoxEnv"
)

register(
    id = "ChaserInvaderDiscreteEnv-v0",
    entry_point = "myenv.env_chaser_invader_discrete:ChaserInvaderDiscreteEnv"
)

register(
    id = "InvaderBoxEnv-v0",
    entry_point = "myenv.env_invader_box:InvaderBoxEnv"
)