@torch.jit.script
def compute_reward(velocity: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate potential-based reward for moving towards the target
    potential_reward = potentials - prev_potentials
    
    # Reward for running faster (forward velocity)
    forward_speed_reward = velocity[:, 0]  # Assuming x-axis corresponds to forward direction
    
    # Penalize high angular velocities to maintain humanoid stability
    penalize_angular_velocity = -torch.norm(ang_velocity, p=2, dim=1)
    
    # Normalization and transformation parameters
    potential_temp = 1.0  # temperature for exponential scaling of potential reward
    speed_temp = 0.1  # temperature for exponential scaling of speed reward
    ang_vel_temp = 0.1  # temperature for exponential scaling of angular velocity penalty
    
    # Applying transformations for each reward component
    transformed_potential_reward = torch.exp(potential_temp * potential_reward) - 1.0
    transformed_forward_speed_reward = torch.exp(speed_temp * forward_speed_reward) - 1.0
    transformed_angular_velocity_penalty = torch.exp(ang_vel_temp * penalize_angular_velocity) - 1.0
    
    # Combining the rewards components
    total_reward = transformed_potential_reward + transformed_forward_speed_reward + transformed_angular_velocity_penalty
    
    # Returning the total reward and individual components
    reward_dict = {
        "potential_reward": transformed_potential_reward,
        "forward_speed_reward": transformed_forward_speed_reward,
        "angular_velocity_penalty": transformed_angular_velocity_penalty,
    }
    
    return total_reward, reward_dict
