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
    
    # Modified Reward for forward velocity with updated temperature
    forward_vel_temperature = 0.15  # Further raise temperature for more influence
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Adjusted Upright reward
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 0.2  # Reduced temperature to deprioritize
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 1.0))

    # Reworked Stability Reward to consider dynamic motion range
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.3
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))  # Penalizing angular movement
    
    # Total reward as a weighted sum of components with updated weights
    total_reward = 1.5 * forward_vel_reward + 0.2 * upright_reward + 0.3 * stability_reward

    # Construct the dictionary for reward components
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
