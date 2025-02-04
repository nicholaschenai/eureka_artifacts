@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Normalize and Reward for forward velocity
    forward_velocity_temp = 0.2  # New temperature for forward velocity
    forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity) - 1

    # Revised penalty for high action magnitudes (to promote efficiency)
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.05  # Adjusted temperature for better scaling
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty) - 0.5

    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled

    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
