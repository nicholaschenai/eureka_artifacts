@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    velocity = root_states[:, 7:10]
    torso_position = root_states[:, 0:3]
    targets = torch.zeros_like(torso_position)  # Assuming targets are at origin for directional consistency

    # Calculate forward velocity
    forward_velocity = velocity[:, 0]  # Assuming x-axis is the forward direction

    # Alignment to running direction component
    direction_to_target = targets - torso_position
    direction_to_target_norm = torch.norm(direction_to_target[:, :2], p=2, dim=-1) + 1e-6
    direction_to_target_normalized = direction_to_target[:, :2] / direction_to_target_norm.unsqueeze(-1)
    forward_dir_vector = velocity[:, :2]
    forward_dir_vector_norm = torch.norm(forward_dir_vector, p=2, dim=-1) + 1e-6
    forward_dir_normalized = forward_dir_vector / forward_dir_vector_norm.unsqueeze(-1)
    cos_angle = torch.sum(forward_dir_normalized * direction_to_target_normalized, dim=-1)

    # Smoothness penalty for actions to encourage efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Assigning Temperature values
    velocity_temperature = 1.0
    angle_alignment_temperature = 0.5
    action_smoothness_temperature = 0.1

    # Reward component calculations
    velocity_reward = torch.exp(velocity_temperature * forward_velocity)
    alignment_reward = torch.exp(angle_alignment_temperature * cos_angle)
    smoothness_penalty = -torch.exp(action_smoothness_temperature * action_penalty)

    # Total reward calculation
    total_reward = velocity_reward + alignment_reward + smoothness_penalty

    # Creating a dictionary of individual reward components
    reward_components = {
        "velocity_reward": velocity_reward,
        "alignment_reward": alignment_reward,
        "smoothness_penalty": smoothness_penalty,
    }

    return total_reward, reward_components
