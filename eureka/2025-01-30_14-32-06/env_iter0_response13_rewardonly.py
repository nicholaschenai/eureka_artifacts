@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information
    velocity = root_states[:, 7:10]
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    
    # Calculate forward velocity component (assuming forward is along the x-axis)
    forward_velocity = velocity[:, 0]
    
    # Align reward with the heading direction
    heading_alignment = torch.sum(heading_vec * torch.tensor([1.0, 0.0, 0.0], device=heading_vec.device), dim=-1)
    heading_temp = 1.0
    heading_reward = torch.exp(heading_temp * heading_alignment) - 1.0

    # Potential-based reward to encourage moving towards the target
    potential_reward = potentials - prev_potentials
    potential_temp = 1.0
    potential_reward_normalized = torch.exp(potential_temp * potential_reward) - 1.0

    # Penalize excessive angular velocity
    ang_velocity = root_states[:, 10:13]
    ang_velocity_penalty = torch.sum(ang_velocity * ang_velocity, dim=-1)
    ang_velocity_temp = 1.0
    angular_penalty = -(torch.exp(ang_velocity_temp * ang_velocity_penalty) - 1.0)
    
    # Total reward combining the components
    reward = forward_velocity + heading_reward + potential_reward_normalized + angular_penalty
    
    # Return the reward and the individual components for analysis
    return reward, {
        "forward_velocity": forward_velocity,
        "heading_reward": heading_reward,
        "potential_reward": potential_reward,
        "angular_penalty": angular_penalty
    }
