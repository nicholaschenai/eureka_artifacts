@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    # We assume forward direction is represented by the x-index, which is 0
    forward_velocity = velocity[:, 0]

    # Reward for moving forward rapidly
    reward_forward_velocity = forward_velocity

    # Penalty for large actions to encourage energy efficiency
    action_penalty_factor = 0.01
    reward_action_penalty = -action_penalty_factor * torch.sum(actions**2, dim=1)

    # Total reward calculation
    total_reward = reward_forward_velocity + reward_action_penalty

    # Compile reward components into a dictionary
    reward_components = {
        "reward_forward_velocity": reward_forward_velocity,
        "reward_action_penalty": reward_action_penalty
    }
    
    return total_reward, reward_components
