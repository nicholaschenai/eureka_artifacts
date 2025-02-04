@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract components from the state
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Forward Velocity Reward
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    forward_vel_temperature = 0.25  # Adjusted for sharper reward
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # New Upright Reward
    upright_temperature = 1.5  # Strong encouragement for consistent posture
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_similarity = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_reward = torch.exp(upright_temperature * (upright_similarity - 1.0))

    # Stability Reward
    stability_temperature = 0.5  # Fine-tuned to promote reduced angular velocities
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Total Reward
    total_reward = 1.0 * forward_vel_reward + 0.3 * upright_reward + 0.2 * stability_reward

    # Reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
