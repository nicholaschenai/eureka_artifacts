@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the current velocity of the humanoid's torso
    velocity = root_states[:, 7:10]  # Assuming (batch_size, 13), where velocity is at position indexes 7 to 9

    # Reward for moving forward faster
    velocity_reward = velocity[:, 0]  # Use the x-component as running direction
    velocity_temp = 0.1
    velocity_reward = torch.exp(velocity_temp * velocity_reward)

    # Penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_temp = 0.01
    action_penalty = torch.exp(-action_penalty_temp * action_penalty)

    # Combine rewards
    total_reward = velocity_reward * action_penalty

    # Return the total reward and individual components for further analysis
    reward_components = {
        "velocity_reward": velocity_reward,
        "action_penalty": action_penalty,
    }

    return total_reward, reward_components
