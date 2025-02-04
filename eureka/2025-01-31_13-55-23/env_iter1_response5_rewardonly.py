@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity
    forward_velocity_reward = forward_velocity
    
    # Re-scaling energy penalty to enhance sensitivity
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Adding a small reward for maintaining a stable torso height to encourage balance
    torso_height = root_states[:, 2]  # Assuming height is the z-axis
    target_height = 0.5  # Hypothetical target height
    height_difference = torch.abs(torso_height - target_height)
    height_temp = 5.0
    height_reward = torch.exp(-height_temp * height_difference)
    
    # Total reward with adjusted scaling
    total_reward = forward_velocity_reward + 0.1 * energy_penalty_scaled + 0.2 * height_reward
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "height_reward": height_reward
    }

    return total_reward, reward_dict
