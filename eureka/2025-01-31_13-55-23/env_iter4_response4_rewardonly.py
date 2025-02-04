@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Enhanced forward velocity reward with increased importance
    forward_velocity_temp = 0.2
    forward_velocity_reward = forward_velocity_temp * forward_velocity

    # Energy penalty with stronger negative impact
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.4
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Modified stable motion component to smoothness encouragement
    smoothness_gain = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    smoothness_temp = 0.3
    smoothness_reward = -smoothness_temp * smoothness_gain

    # Total reward calculated using temperature scaling
    overall_temp = 0.15
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + smoothness_reward))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "smoothness_reward": smoothness_reward
    }

    return total_reward, reward_dict
