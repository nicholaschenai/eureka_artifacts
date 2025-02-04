@torch.jit.script
def compute_reward(velocity: torch.Tensor, ang_velocity: torch.Tensor, torso_position: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for transformation functions
    velocity_temp = 0.1
    ang_velocity_temp = 0.1
    upright_temp = 0.1

    # Compute components
    # Reward for forward velocity - encouraging higher speed
    forward_velocity = velocity[:, 0]
    reward_velocity = forward_velocity

    # Penalty for angular velocity - discouraging excessive spinning
    ang_velocity_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    transformed_ang_velocity_penalty = -torch.exp(-ang_velocity_penalty / ang_velocity_temp)

    # Reward for remaining upright - encouraging upright posture
    upright_bonus = 1.0 - torch.abs(torso_position[:, 2] - 1.0)  # Assuming desired height is 1.0
    transformed_upright_bonus = torch.exp(upright_bonus * upright_temp)

    # Combine rewards
    total_reward = reward_velocity + transformed_ang_velocity_penalty + transformed_upright_bonus

    # Dictionary of individual reward components
    reward_components = {
        "reward_velocity": reward_velocity,
        "transformed_ang_velocity_penalty": transformed_ang_velocity_penalty,
        "transformed_upright_bonus": transformed_upright_bonus,
    }

    return total_reward, reward_components
