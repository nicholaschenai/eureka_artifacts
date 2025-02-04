@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Re-scaled Forward Velocity Reward
    forward_vel_temperature = 0.1  # Updated temperature for better scaling
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Reworked Upright Reward: Penalize large deviation
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_penalty_temperature = 2.0  # More sensitive transformation to deviations
    upright_penalty = torch.exp(-upright_penalty_temperature * (1.0 - dot_prod_up))

    # Total reward
    total_reward = 2.0 * forward_vel_reward + 1.0 * upright_penalty

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_penalty": upright_penalty
    }

    return total_reward, reward_dict
