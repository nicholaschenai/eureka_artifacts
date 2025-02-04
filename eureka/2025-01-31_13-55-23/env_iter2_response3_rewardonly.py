@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Slightly increased weight for forward velocity to enhance its significance
    velocity_scale = 1.0
    forward_velocity_reward = velocity_scale * forward_velocity
    
    # Enhanced energy penalty to be more sensitive to high action magnitudes
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Increased from 0.5 to 1.0 to increase sensitivity
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # New smoothness component for smoother actions over consecutive timesteps
    action_smoothness_temp = 0.5
    action_diff = actions[:, 1:] - actions[:, :-1]
    action_smoothness_penalty = torch.exp(-action_smoothness_temp * torch.sum(action_diff**2, dim=-1))

    # Total reward, now includes action smoothness
    total_reward = forward_velocity_reward + energy_penalty_scaled + action_smoothness_penalty
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "action_smoothness_penalty": action_smoothness_penalty
    }

    return total_reward, reward_dict
