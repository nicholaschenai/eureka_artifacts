@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Scale the forward velocity reward 
    forward_temp = 0.5
    forward_velocity_reward = torch.exp(forward_temp * forward_velocity) - 1.0

    # Redesigned penalty for high action magnitudes (more sensitive)
    action_magnitude = torch.sum(actions**2, dim=-1)
    energy_temp = 0.2  # Adjusted temperature parameter for better sensitivity
    energy_penalty_scaled = torch.exp(-energy_temp * action_magnitude)

    # Total reward: combining with adjusted weights to balance components
    total_temp = 1.0
    total_reward = (forward_velocity_reward + energy_penalty_scaled) * total_temp

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
