@torch.jit.script
def compute_reward(root_states: torch.Tensor, velocity: torch.Tensor, ang_velocity: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity component (assuming y-axis is forward)
    forward_velocity = velocity[:, 1]
    
    # Reward proportional to forward velocity
    forward_velocity_reward = forward_velocity

    # Penalize high angular velocity (to maintain stability)
    angular_velocity_penalty = torch.norm(ang_velocity, p=2, dim=-1)

    # Penalize excessive actions to discourage large, inefficient movements
    action_penalty = torch.norm(actions, p=2, dim=-1)

    # Calculate total reward
    total_reward = forward_velocity_reward - 0.1 * angular_velocity_penalty - 0.01 * action_penalty
    
    # Normalize and transform if needed
    temperature_velocity = 0.1
    reward_transformed = torch.exp(forward_velocity_reward / temperature_velocity) - 1
    
    # Add the transformed reward to the total reward
    total_reward += reward_transformed
    
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
        "action_penalty": action_penalty,
        "reward_transformed": reward_transformed
    }
    
    return total_reward, reward_components
