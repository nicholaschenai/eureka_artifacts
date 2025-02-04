@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary parameters
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Increase Forward Velocity Reward's emphasis
    forward_vel_temperature = 0.15  # Adjusted temperature for enhanced influence
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Moderate Upright Reward's impact 
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 0.1  # Reduced leverage for balancing
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 1.0))

    # Introduce Energy Efficiency Reward
    # Penalize the magnitude of velocity for efficiency
    energy_efficiency_temperature = 0.2
    energy_efficiency_reward = torch.exp(-energy_efficiency_temperature * torch.norm(velocity, p=2, dim=-1))

    # Total reward emphasizing forward velocity, moderate posture, and efficiency
    total_reward = 1.5 * forward_vel_reward + 0.2 * upright_reward + 0.3 * energy_efficiency_reward

    # Reward dictionary construction
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "energy_efficiency_reward": energy_efficiency_reward
    }

    return total_reward, reward_dict
