@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Re-scaled range for further movement incentive
    forward_speed_max = 15.0
    forward_reward = forward_velocity.clamp(min=0) / forward_speed_max

    # Recalibrated sideways penalty
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    revised_sideways_penalty = -0.3 * sideways_velocity / forward_speed_max

    # New component: Smoothness Reward
    action_smoothness = -torch.sum(torch.abs(actions[:, 1:] - actions[:, :-1]), dim=-1)
    smoothness_penalty = 0.01 * action_smoothness

    # Transform reward components
    temperature_forward = 1.2
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Amplify sideways impact in the penalty
    temperature_sideways = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * revised_sideways_penalty) - 1.0

    # Total reward calculation
    combined_reward = transformed_forward_reward + transformed_sideways_penalty + smoothness_penalty

    return combined_reward, {
        "forward_reward": forward_reward,
        "revised_sideways_penalty": revised_sideways_penalty,
        "smoothness_penalty": smoothness_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
