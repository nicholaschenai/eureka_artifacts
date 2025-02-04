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
    
    # Forward Velocity Reward - slightly reduced influence
    forward_vel_temperature = 0.1  # Reduced to balance other rewards
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Redesigned Upright Reward - encourage upright with broader range
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_reward_scale = 0.5  # Increase scaling to enhance effect
    upright_reward = torch.clamp(dot_prod_up, min=0.5) * upright_reward_scale

    # Stability Reward - transform for finer grading
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 2.0  # Increased for finer differentiation
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Total reward with revised weightings
    total_reward = 0.8 * forward_vel_reward + 0.7 * upright_reward + 0.5 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
