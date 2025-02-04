@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Rescaling the forward velocity reward for better balance
    forward_velocity_scale = 1.3  # Slightly reduced from 1.5
    forward_velocity_reward = forward_velocity_scale * forward_velocity
    
    # Enhance energy penalty's influence
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Adjusting the temperature for stronger penalization
    energy_temp = 1.2  # Stronger penalization
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)

    # Aggregate the rewards
    total_temp = 0.1  # Normalizing both rewards to ensure competitive balance
    total_reward = torch.exp(total_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
