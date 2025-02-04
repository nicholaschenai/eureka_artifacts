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

    # Rescale Forward Velocity Reward
    forward_vel_temperature = 0.1  # Reduced to balance contribution
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Reformulated Upright Reward considering deviation angle
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.clamp(torch.sum(up_vec * up_vector_expected, dim=-1), -1.0, 1.0)
    upright_angle = torch.acos(dot_prod_up)  # Angle deviation from straight up
    upright_temperature = -2.0  # Encourage uprightness by penalizing larger angles
    upright_reward = torch.exp(upright_temperature * upright_angle) - 1.0

    # Modified Stability Reward with adjusted temperature
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.5  # Amplify to improve differentiation
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Total reward with refined contribution ratios
    total_reward = 0.5 * forward_vel_reward + 0.25 * upright_reward + 0.25 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
