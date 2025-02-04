@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Forward Velocity Reward
    forward_vel_temperature = 0.15
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Angle deviation from upright
    upright_threshold = 0.1  # Allow for a minor deviation
    deviation_angle = torch.abs(torch.acos(torch.clamp(up_vec[:, 2], -1.0, 1.0)))
    upright_temperature = 0.5
    upright_reward = torch.exp(-upright_temperature * deviation_angle) - 1.0

    # Stability Reward
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.2
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Total reward
    total_reward = 1.5 * forward_vel_reward + 0.4 * upright_reward + 0.6 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
