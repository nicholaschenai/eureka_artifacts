@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract position and velocity components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate directional velocity towards the target
    to_target_dir = targets - torso_position
    to_target_dir[:, 2] = 0.0
    to_target_norm = torch.nn.functional.normalize(to_target_dir, dim=-1)
    forward_velocity = torch.sum(velocity * to_target_norm, dim=-1)
    
    # Adjusted forward velocity reward
    forward_velocity_temperature = 0.1  # Enhancing temperature for better scaling
    forward_velocity_reward = torch.exp(forward_velocity_temperature * forward_velocity) - 1.0

    # New formulation for upright reward using deviation penalty
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    deviation_penalty = torch.pow(dot_prod_up - 1.0, 2)
    
    upright_temperature = 0.5  # Chosen suitable temperature for good differentiation
    upright_reward = torch.exp(-upright_temperature * deviation_penalty)

    # Total reward combining both components
    total_reward = forward_velocity_reward + upright_reward * 0.5

    # Prepare the dictionary of individual rewards
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "upright_reward": upright_reward
    }

    return total_reward, reward_dict
