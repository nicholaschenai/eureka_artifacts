@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming forward is along the x-axis)
    forward_velocity = root_states[:, 7]  # x-axis component of velocity
    
    # Reward for moving forward (higher velocity gets higher reward)
    forward_reward = forward_velocity * 10.0  # Scale the forward reward
    
    # Reward for reducing the distance to the target
    distance_reward = potentials - prev_potentials
    distance_temperature = 10.0
    distance_reward_transformed = torch.exp(distance_temperature * distance_reward)
    
    # Penalty for excessive actions (to encourage efficient movements)
    action_penalty = torch.sum(actions**2, dim=-1)
    action_temperature = 0.01
    action_penalty_transformed = torch.exp(-action_temperature * action_penalty)
    
    # Combine rewards
    total_reward = forward_reward + distance_reward_transformed - action_penalty_transformed
    
    # Return total reward and individual components
    reward_components = {
        "forward_reward": forward_reward,
        "distance_reward_transformed": distance_reward_transformed,
        "action_penalty_transformed": action_penalty_transformed
    }
    
    return total_reward, reward_components
