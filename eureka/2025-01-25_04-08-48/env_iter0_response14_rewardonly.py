@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity (assuming y-axis is forward direction)
    forward_velocity = root_states[:, 7]

    # Reward for moving fast in the forward direction
    forward_velocity_reward = forward_velocity

    # Penalty for deviating from the upright position
    up_penalty_temp = 0.5
    up_penalty = torch.exp(-up_penalty_temp * (1.0 - up_vec[:, 2]))  # Encourage the up_vec's z component to be close to 1

    # Penalty for high angular velocities (to avoid unnecessary rotation)
    ang_velocity_penalty_temp = 1.0
    ang_velocity_penalty = torch.exp(-ang_velocity_penalty_temp * torch.norm(ang_velocity, p=2, dim=-1))

    # Total reward calculation, weighted sum
    total_reward = forward_velocity_reward + up_penalty + ang_velocity_penalty

    # Normalize the total reward
    total_reward = (total_reward - total_reward.min()) / (total_reward.max() - total_reward.min())

    # Reward components dictionary
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "up_penalty": up_penalty,
        "ang_velocity_penalty": ang_velocity_penalty,
    }

    return total_reward, reward_components
