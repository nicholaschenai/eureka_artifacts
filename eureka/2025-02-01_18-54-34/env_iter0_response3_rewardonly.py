@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming it's the first component of velocity vector)
    forward_velocity = root_states[:, 7]  # Assuming x-axis is the forward direction
    
    # Define a temperature for normalizing the velocity-based reward
    velocity_reward_temperature = 0.1
    
    # Compute the velocity-based reward
    velocity_reward = torch.exp(velocity_reward_temperature * forward_velocity)
    
    # Penalize the joint torques slightly to encourage energy efficiency
    torque_penalty_temperature = -0.01
    torque_penalty = torch.exp(torque_penalty_temperature * torch.norm(dof_force_tensor, p=2, dim=-1))
    
    # Compute the total reward
    total_reward = velocity_reward + torque_penalty
    
    # Create dictionary of individual reward components
    reward_dict = {
        "velocity_reward": velocity_reward,
        "torque_penalty": torque_penalty,
    }
    
    return total_reward, reward_dict
