@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor, sensor_force_torques: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary variables
    velocity = root_states[:, 7:10]
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]

    # Calculate the forward velocity component (assuming forward is along the x-axis)
    forward_velocity = velocity[:, 0]

    # Calculate direction alignment with the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    direction_alignment = torch.cos(torch.atan2(to_target[:, 1], to_target[:, 0]) - torch.atan2(velocity[:, 1], velocity[:, 0]))

    # Penalty for collision forces
    contact_penalty = torch.sum(sensor_force_torques**2, dim=1)

    # Penalty on actions to encourage smoother movements
    action_penalty = torch.sum(actions**2, dim=1)

    # Reward components
    velocity_reward = forward_velocity
    alignment_reward = direction_alignment

    # Transform rewards and penalties if needed
    # Use temperature to scale rewards and penalties
    velocity_temperature = 0.1
    alignment_temperature = 0.1
    contact_temperature = 0.1
    action_temperature = 0.1

    velocity_reward_transformed = torch.exp(velocity_reward / velocity_temperature)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temperature)
    contact_penalty_transformed = -torch.exp(contact_penalty / contact_temperature)
    action_penalty_transformed = -torch.exp(action_penalty / action_temperature)

    # Total reward
    total_reward = velocity_reward_transformed + alignment_reward_transformed + contact_penalty_transformed + action_penalty_transformed

    # Return the computed reward and its components
    reward_components = {
        "velocity_reward": velocity_reward_transformed,
        "alignment_reward": alignment_reward_transformed,
        "contact_penalty": contact_penalty_transformed,
        "action_penalty": action_penalty_transformed,
    }

    return total_reward, reward_components
