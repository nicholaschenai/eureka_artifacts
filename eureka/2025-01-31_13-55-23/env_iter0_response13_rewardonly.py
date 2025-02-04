@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information
    velocity = root_states[:, 7:10]  # Extract [vx, vy, vz] velocities
    ang_velocity = root_states[:, 10:13]  # Extract angular velocities
    
    # Define temperature parameters for reward transformations
    velocity_temp = 1.0
    stability_temp = 1.0
    control_effort_temp = 0.01
    
    # Forward velocity (maximize this component)
    forward_velocity_reward = velocity[:, 0]  # vx represents forward direction
    forward_velocity_reward_transformed = torch.exp(forward_velocity_reward / velocity_temp)
    
    # Stability penalty (minimize angular velocities for stability)
    angular_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    stability_penalty_transformed = torch.exp(-angular_penalty / stability_temp)
    
    # Control effort penalty (reduce the magnitude of actions to minimize energy consumption)
    control_effort = torch.norm(actions, p=2, dim=-1)
    control_effort_penalty_transformed = torch.exp(-control_effort / control_effort_temp)

    # Total reward as a weighted sum of components
    total_reward = forward_velocity_reward_transformed + stability_penalty_transformed + control_effort_penalty_transformed
    
    # Organizing each reward component in a dictionary
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward_transformed,
        "stability_penalty": stability_penalty_transformed,
        "control_effort_penalty": control_effort_penalty_transformed
    }

    return total_reward, reward_components
