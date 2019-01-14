import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FourRooms-v0',
    entry_point='rooms.envs:FourRooms',
    timestep_limit=10000,
    reward_threshold=1.0,
    nondeterministic = True,
)