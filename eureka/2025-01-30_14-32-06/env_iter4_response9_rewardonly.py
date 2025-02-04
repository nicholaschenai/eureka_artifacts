@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Forward Velocity Reward
    forward_vel_temperature = 0.1  # Adjust temperature for better scaling
    forward_vel_reward = torch.tanh(forward_vel_temperature * forward_vel)

    # Upright Reward with adjusted scaling
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 1.0  # Increase influence to be more substantial
    upright_reward = torch.tanh(upright_temperature * (dot_prod_up - 0.5))  # Tighter tolerance now seen as ideal

    # Refined Stability Reward
    ang_velocity = root_states[:, 10:13]
    smoothness_temperature = 0.5  # Changed to promote more varied responses
    stability_reward = torch.tanh(smoothness_temperature * (1.0 - torch.norm(ang_velocity, p=2, dim=-1)))

    # Total reward with recalibrated weightings
    total_reward = 1.5 * forward_vel_reward + 1.0 * upright_reward + 0.5 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
