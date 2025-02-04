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
    
    # Adjusted Reward for Forward Velocity using a temperature for better range
    forward_vel_temperature = 0.1  # Reduced temperature to mitigate overpowering
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Upright Reward - modified for variance and relevance
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 0.5  # Increasing to enhance influential differentiation
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 1.0))  # Use exp to enlarge variance

    # Re-defined Stability Reward to curb rapid deviations
    ang_velocity = root_states[:, 10:13]
    smoothness_temperature = 0.3  # Higher temperature for considerable influence 
    stability_reward = -torch.exp(smoothness_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Total reward normalization
    total_reward = 1.0 * forward_vel_reward + 0.4 * upright_reward + 0.6 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
