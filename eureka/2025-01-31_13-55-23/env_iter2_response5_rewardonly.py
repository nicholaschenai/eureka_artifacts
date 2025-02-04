@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Measure of the forward velocity, with temperature scaling
    velocity_temp = 0.1
    forward_velocity_reward = forward_velocity
    
    # Energy penalty, increased temperature for stronger penalization
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Increased from 0.5 for stronger penalization
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward
    total_reward = torch.exp(velocity_temp * forward_velocity_reward) + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
