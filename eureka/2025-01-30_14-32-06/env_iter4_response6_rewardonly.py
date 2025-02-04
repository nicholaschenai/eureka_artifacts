@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Penalize large deviation from the optimal forward direction
    forward_vel_temperature = 0.3  # Enhanced for sharper discrimination at higher velocites
    forward_vel_reward = torch.clamp(torch.exp(forward_vel_temperature * (forward_vel - 5.0)) - 1.0, min=0.0)
    
    # Upright adjustment
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_penalty_temperature = 0.5  # Convert to penalty for deviation from upright
    upright_reward = torch.exp(-upright_penalty_temperature * (1.0 - dot_prod_up))
    
    # Modified Stability Reward
    ang_velocity = root_states[:, 10:13]
    smoothness_penalty_temperature = 0.25  # Adjusted to make medium refinements count
    stability_reward = torch.exp(-smoothness_penalty_temperature * torch.norm(ang_velocity, p=2, dim=-1))
    
    # Total reward aggregation
    total_reward = 1.0 * forward_vel_reward + 0.1 * upright_reward + 0.2 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
