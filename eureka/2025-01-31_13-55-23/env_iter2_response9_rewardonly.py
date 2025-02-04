@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assume forward direction is along the x-axis

    # Reward for forward velocity with transformation
    velocity_temp = 0.1  # Temperature for velocity transformation
    forward_velocity_reward = torch.exp(velocity_temp * forward_velocity) - 1

    # Energy penalty adjusted for sensitivity
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.2  # Increased temperature for energy penalty
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)

    # Balance total reward
    total_reward = forward_velocity_reward * 0.5 + energy_penalty_scaled * 0.5
    
    # Individual reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
