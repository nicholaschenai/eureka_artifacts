@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Front velocity
    forward_velocity = root_states[:, 7]  # Index 7 is the forward (x-axis) velocity component
    
    # Reward based on forward velocity
    forward_reward = forward_velocity
    
    # Action regularization term to encourage minimal effort
    action_penalty = torch.sum(actions ** 2, dim=1)
    action_penalty_weight = 0.1
    
    # Total reward
    total_reward = forward_reward - action_penalty_weight * action_penalty
    
    # Transforming the rewards (optional)
    forward_vel_temperature = 1.0
    action_penalty_temperature = 1.0

    transformed_forward_reward = torch.exp(forward_reward / forward_vel_temperature)
    transformed_action_penalty = -torch.exp(-action_penalty / action_penalty_temperature)
    
    total_transformed_reward = transformed_forward_reward + transformed_action_penalty

    # Prepare the dictionary of reward components
    reward_components = {
        "forward_reward": forward_reward,
        "action_penalty": action_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_action_penalty": transformed_action_penalty
    }
    
    return total_transformed_reward, reward_components
