@torch.jit.script
def compute_reward(root_states: torch.Tensor, 
                   actions: torch.Tensor, 
                   up_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Getting the forward velocity along the x-axis
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Assuming the x-axis is the forward direction

    # Reward for going forward
    forward_reward = forward_velocity

    # Penalty for large or unnecessary actions (encourages smoothness)
    action_penalty_factor = 0.01
    action_penalty = action_penalty_factor * torch.sum(torch.abs(actions), dim=-1)

    # Reward for maintaining balance, i.e., having a high up-projection on the z-axis
    balance_reward_factor = 0.1
    balance_reward = balance_reward_factor * (1.0 - torch.abs(1.0 - up_proj))

    # Total reward, balancing all components
    total_reward = forward_reward - action_penalty + balance_reward

    # Temperature parameters for each reward component
    forward_reward_temperature = 1.0
    action_penalty_temperature = 1.0
    balance_reward_temperature = 1.0

    # Apply temperature based exponential transformation for each component
    forward_reward = torch.exp(forward_reward_temperature * forward_reward)
    action_penalty = torch.exp(action_penalty_temperature * (-action_penalty))
    balance_reward = torch.exp(balance_reward_temperature * balance_reward)

    # Recalculate total reward after transformation
    transformed_total_reward = forward_reward + balance_reward - action_penalty

    # Create dictionary for each reward component
    reward_dict = {
        "forward_reward": forward_reward,
        "action_penalty": -action_penalty,
        "balance_reward": balance_reward
    }

    return transformed_total_reward, reward_dict
