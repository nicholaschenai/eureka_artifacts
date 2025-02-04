@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Reward for forward velocity
    forward_velocity_reward = 0.5 * forward_velocity
    
    # More pronounced penalty for high action magnitudes
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Elevating the penalty using a higher temperature
    energy_temp = 1.0  # Increased for stronger penalization
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)

    # Reward for longer episode duration (normalized)
    max_episode_length = 1000  # Assume max episode length
    duration_bonus_temp = 0.01  # A small temperature for gentle scaling
    duration_bonus = torch.exp(duration_bonus_temp * episode_length / max_episode_length)

    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled + duration_bonus
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "duration_bonus": duration_bonus
    }

    return total_reward, reward_dict
