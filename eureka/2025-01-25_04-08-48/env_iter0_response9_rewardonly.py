@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, contact_force_scale: float, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from the root states
    velocity = root_states[:, 7:10]
    torso_height = root_states[:, 2]

    # Reward for forward velocity
    forward_velocity_reward = velocity[:, 0]  # Assuming the forward direction is the x-axis

    # Penalty for large actions to encourage smooth movements
    action_penalty = torch.sum(actions**2, dim=-1)

    # Stability bonus for maintaining a certain torso height
    desired_height = torch.tensor(1.0, device=root_states.device)  # Example desired torso height
    height_penalty = torch.abs(torso_height - desired_height)

    # Combine the rewards and penalties
    total_reward = forward_velocity_reward - 0.01 * action_penalty - 0.5 * height_penalty

    # Normalization and temperature parameters
    forward_temperature = 1.0
    action_temp = 0.1
    height_temp = 0.2

    # Applying transformations
    forward_velocity_reward = torch.exp(forward_velocity_reward / forward_temperature)
    action_penalty = torch.exp(-action_penalty / action_temp)
    height_penalty = torch.exp(-height_penalty / height_temp)

    reward_dict = {
        'forward_velocity_reward': forward_velocity_reward,
        'action_penalty': -0.01 * action_penalty,
        'height_penalty': -0.5 * height_penalty
    }

    total_reward = forward_velocity_reward - 0.01 * action_penalty - 0.5 * height_penalty

    return total_reward, reward_dict
