import numpy as np
from enums import LakeState


LAKE_SMALL = np.array([[LakeState.SAFE, LakeState.FROZEN, LakeState.FROZEN, LakeState.FROZEN],
                       [LakeState.FROZEN, LakeState.HOLE, LakeState.FROZEN, LakeState.HOLE],
                       [LakeState.FROZEN, LakeState.FROZEN, LakeState.FROZEN, LakeState.HOLE],
                       [LakeState.HOLE, LakeState.FROZEN, LakeState.FROZEN, LakeState.GOAL]])


REWARD_MAP = {LakeState.SAFE: 0,
              LakeState.FROZEN: 0,
              LakeState.HOLE: -1,
              LakeState.GOAL: 1}
