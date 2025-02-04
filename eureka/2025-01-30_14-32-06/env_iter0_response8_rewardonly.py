@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity of the Ant
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Calculate the change in potential which correlates with moving towards the target
    progress_reward = potentials - prev_potentials

    # Encourage higher forward speed
    forward_speed_reward = forward_velocity

    # Penalize excessive actions to encourage efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Normalize rewards for stability
    temperature_forward_speed = 0.1
    temperature_action_penalty = 0.01

    normalized_forward_speed_reward = torch.exp(forward_speed_reward / temperature_forward_speed)
    normalized_action_penalty = torch.exp(-action_penalty / temperature_action_penalty)

    # Total reward combines incentives to move fast and penalize excessive actions
    total_reward = progress_reward + normalized_forward_speed_reward - normalized_action_penalty

    # Compile the components of the reward
    reward_components = {
        "progress_reward": progress_reward,
        "forward_speed_reward": forward_speed_reward,
        "normalized_forward_speed_reward": normalized_forward_speed_reward,
        "action_penalty": action_penalty,
        "normalized_action_penalty": normalized_action_penalty,
    }

    return total_reward, reward_components
