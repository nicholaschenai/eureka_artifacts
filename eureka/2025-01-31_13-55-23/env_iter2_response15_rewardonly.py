@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Linear velocity extraction
    forward_velocity = velocity[:, 0]  # Forward direction velocity (x-axis assumed)
    
    # Forward velocity reward with adjusted scale
    velocity_scale = 1.0  # Increased to emphasize forward movement
    forward_velocity_reward = velocity_scale * forward_velocity

    # Adjusted energy penalty with more focus
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Increased from 0.5 to 1.0 to provide stronger penalties for actions
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)

    # Adjust reward mixing to balance the reward components
    total_reward = forward_velocity_reward + 0.5 * energy_penalty_scaled
    
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
