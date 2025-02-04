@torch.jit.script
def compute_reward(root_states: torch.Tensor, velocity: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward components parameters
    forward_velocity_weight = 1.0
    heading_alignment_weight = 0.5
    angular_velocity_penalty_weight = -0.1
    
    # Extract forward velocity (assuming forward is along x-axis)
    forward_velocity = velocity[:, 0]
    
    # Compute reward for forward velocity
    forward_velocity_reward = forward_velocity * forward_velocity_weight
    
    # Normalize heading vector to x-axis
    desired_heading_vec = torch.tensor([1.0, 0.0, 0.0], device=root_states.device)
    heading_alignment_reward = torch.sum(heading_vec * desired_heading_vec, dim=-1) * heading_alignment_weight
    
    # Penalty for excessive angular velocity (stability component)
    angular_velocity = root_states[:, 10:13]
    angular_velocity_penalty = torch.norm(angular_velocity, p=2, dim=-1) * angular_velocity_penalty_weight
    
    # Total reward
    total_reward = forward_velocity_reward + heading_alignment_reward + angular_velocity_penalty
    
    # Return total reward and individual components
    rewards_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "heading_alignment_reward": heading_alignment_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
    }
    
    return total_reward, rewards_dict
