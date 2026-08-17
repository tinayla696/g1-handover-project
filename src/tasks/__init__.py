import gymnasium as gym
from .g1_handover_env import (
    G1HandoverEnvCfg,
    G1HandoverPPORunnerCfg,
    G1HandoverResidualKinematicEnvCfg,
    G1HandoverResidualKinematicPPORunnerCfg,
)

gym.register(
    id="G1-Handover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "cfg": G1HandoverEnvCfg(),
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_handover_env:G1HandoverPPORunnerCfg",
    },
)

gym.register(
    id="G1-Handover-Residual-Kinematic-v0",
    entry_point="tasks.g1_handover_env:G1HandoverResidualKinematicEnv",
    kwargs={
        "cfg": G1HandoverResidualKinematicEnvCfg(),
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_handover_env:G1HandoverResidualKinematicPPORunnerCfg",
    },
)