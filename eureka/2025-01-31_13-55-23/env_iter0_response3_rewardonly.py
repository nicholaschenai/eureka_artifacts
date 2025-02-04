@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float, prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from `root_states`
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate to_target vector ignoring the height dimension
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    
    # Calculate the current potential based on distance to target
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    
    # Reward components
    forward_vel_reward = velocity[:, 0]  # Reward for forward velocity
    progress_reward = potentials - prev_potentials  # Reward for progressing towards target
    stability_reward = -torch.abs(velocity[:, 1])  # Penalty for lateral velocity to maintain stability
    
    # Temperature parameters for each transformation
    temperature_forward = 1.0  # Only a placeholder, adjust based on experimentation
    temperature_progress = 1.0  # Only a placeholder, adjust based on experimentation
    temperature_stability = 0.5  # Only a placeholder, adjust based on experimentation
    
    # Exponential transformations with respective temperatures
    forward_vel_reward_trans = torch.exp(forward_vel_reward / temperature_forward)
    progress_reward_trans = torch.exp(progress_reward / temperature_progress)
    stability_reward_trans = torch.exp(stability_reward / temperature_stability)
    
    # Total reward as a weighted sum of components
    total_reward = forward_vel_reward_trans + progress_reward_trans + stability_reward_trans
    
    # Return total reward and components
    return total_reward, {
        "forward_vel_reward": forward_vel_reward_trans,
        "progress_reward": progress_reward_trans,
        "stability_reward": stability_reward_trans,
    }
