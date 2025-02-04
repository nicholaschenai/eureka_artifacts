@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Reward for forward velocity further normalized
    velocity_scale = 0.4  # Reduced slightly to balance overall reward components
    forward_velocity_reward = velocity_scale * forward_velocity

    # Further Enhanced penalty for high action magnitudes
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Further increasing the temperature for the energy penalty for stronger penalization
    energy_temp = 1.0  # Increased to 1.0 for even stronger sensitivity to high action magnitudes
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)

    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
