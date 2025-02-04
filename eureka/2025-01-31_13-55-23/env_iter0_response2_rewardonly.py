@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    velocities: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Assuming the forward direction is along the x-axis
    forward_velocity = velocities[:, 0]  # Extract x-component of the velocity

    # Reward for running forward
    forward_reward = forward_velocity

    # Regularization on actions to minimize energy usage
    action_effort_penalty = actions.pow(2).sum(dim=-1)
    
    # Stability term: incentivize being upright
    stability_reward = up_vec[:, 2]  # Higher up axis implies more upright (assuming z is up, adjust accordingly)

    # Heading alignment (encourage heading vector to align with target)
    heading_alignment_reward = torch.dot(heading_vec, torch.unsqueeze(targets, 1))

    # Normalization and temperature parameters.
    forward_velocity_temp = 1.0
    action_effort_temp = 0.01
    stability_temp = 0.1
    heading_alignment_temp = 0.1
    
    # Using soft maximum capacity with exp to normalize and tune components
    forward_reward = torch.exp(forward_reward / forward_velocity_temp)
    action_effort_penalty = -torch.exp(action_effort_penalty * action_effort_temp)
    stability_reward = torch.exp(stability_reward / stability_temp)
    heading_alignment_reward = torch.exp(heading_alignment_reward / heading_alignment_temp)
    
    # Weighted sum of rewards
    total_reward = (
        forward_reward 
        + action_effort_penalty
        + stability_reward
        + heading_alignment_reward
    )

    reward_components = {
        "forward_reward": forward_reward,
        "action_effort_penalty": action_effort_penalty,
        "stability_reward": stability_reward,
        "heading_alignment_reward": heading_alignment_reward
    }

    return total_reward, reward_components
