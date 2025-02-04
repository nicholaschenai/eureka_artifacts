@torch.jit.script
def compute_reward(
    velocity: torch.Tensor, # Velocity of the humanoid
    heading_proj: torch.Tensor, # Projection of heading direction
    actions: torch.Tensor, # Actions taken by humanoid
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Reward for forward velocity to encourage running
    velocity_reward_weight = 1.0
    velocity_reward = velocity[:, 0] * velocity_reward_weight
    
    # Reward for maintaining the correct heading direction
    heading_reward_weight = 0.5
    heading_reward = heading_proj * heading_reward_weight
    
    # Penalty for excessive actions to encourage smooth movements
    action_penalty_weight = 0.01
    action_penalty = torch.sum(actions**2, dim=1) * action_penalty_weight
    
    # Combine all reward components
    total_reward = velocity_reward + heading_reward - action_penalty
    
    return total_reward, {
        "velocity_reward": velocity_reward,
        "heading_reward": heading_reward,
        "action_penalty": action_penalty
    }
