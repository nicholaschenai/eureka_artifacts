@torch.jit.script
def compute_reward(velocity: torch.Tensor, to_target: torch.Tensor, actions: torch.Tensor, rew_scale: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward components
    target_speed = torch.norm(velocity[:, :2], p=2, dim=1)  # Forward speed
    efficiency_penalty = torch.sum(actions**2, dim=1)  # Penalize high energy usage using the L2 norm
    heading_bonus = torch.cos(torch.atan2(to_target[:, 1], to_target[:, 0]))  # Bonus for heading towards the target

    # Define normalization temperature for each reward component
    target_speed_temp = 1.0
    efficiency_penalty_temp = 1.0
    heading_bonus_temp = 1.0

    # Transform reward components
    transformed_target_speed = torch.exp(target_speed / target_speed_temp)
    transformed_efficiency_penalty = torch.exp(-efficiency_penalty / efficiency_penalty_temp)
    transformed_heading_bonus = torch.exp(heading_bonus / heading_bonus_temp)

    # Calculate the total reward
    total_reward = rew_scale * (transformed_target_speed + transformed_heading_bonus - transformed_efficiency_penalty)

    # Return the total reward and its components
    reward_components = {
        "transformed_target_speed": transformed_target_speed,
        "transformed_efficiency_penalty": transformed_efficiency_penalty,
        "transformed_heading_bonus": transformed_heading_bonus,
    }
    
    return total_reward, reward_components
