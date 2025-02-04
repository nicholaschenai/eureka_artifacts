@torch.jit.script
def compute_reward(velocity: torch.Tensor, actions: torch.Tensor, up_proj: torch.Tensor, heading_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component weights
    velocity_weight: float = 1.0
    effort_penalty_weight: float = 0.005
    orientation_penalty_weight: float = 0.1
    heading_reward_weight: float = 0.5

    # Velocity Reward
    forward_velocity: torch.Tensor = velocity[:, 0]  # Assuming forward velocity is along the x-axis
    velocity_reward: torch.Tensor = forward_velocity * velocity_weight

    # Action Effort Penalty
    action_effort_penalty: torch.Tensor = torch.sum(actions ** 2, dim=-1) * effort_penalty_weight

    # Orientation Penalty (penalizing deviation from upright position)
    up_penalty: torch.Tensor = (1.0 - up_proj) * orientation_penalty_weight

    # Heading Reward (aligned with the target direction)
    heading_reward: torch.Tensor = heading_proj * heading_reward_weight

    # Total Reward
    total_reward: torch.Tensor = velocity_reward - action_effort_penalty - up_penalty + heading_reward

    # Reward components dictionary
    reward_components = {
        'velocity_reward': velocity_reward,
        'action_effort_penalty': -action_effort_penalty,
        'up_penalty': -up_penalty,
        'heading_reward': heading_reward
    }

    return total_reward, reward_components
