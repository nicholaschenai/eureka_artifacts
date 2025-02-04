@torch.jit.script
def compute_reward(velocity: torch.Tensor, actions: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for forward velocity
    forward_velocity_reward_temperature = 1.0
    forward_velocity_reward = velocity[:, 0]  # Assuming forward direction is along the x-axis
    forward_velocity_reward_transformed = torch.exp(forward_velocity_reward_temperature * forward_velocity_reward)

    # Penalty for high angular velocity (to promote stability)
    angular_velocity_penalty_temperature = -0.5
    angular_velocity_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    angular_velocity_penalty_transformed = torch.exp(angular_velocity_penalty_temperature * angular_velocity_penalty)

    # Total reward is the sum of the velocity reward and the penalty
    total_reward = forward_velocity_reward_transformed + angular_velocity_penalty_transformed

    # Components of the reward function
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward_transformed,
        "angular_velocity_penalty": angular_velocity_penalty_transformed
    }

    return total_reward, reward_components
