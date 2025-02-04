@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for moving forward; increase max speed for normalization
    max_speed = 5.0  # reduced to focus on reaching lower target quickly and improving faster
    forward_reward = (forward_velocity / max_speed).clamp(-1.0, 1.0)

    # Penalize sideways and backward movement with increased penalty
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * sideways_velocity / max_speed

    # Potentially update or remove heading_reward since it's redundant in this feedback

    # Transform rewards to handle stability better
    temperature1 = 1.0  # Increased to accentuate forward progress
    temperature2 = 0.8  # Adjusted to increase discouragement of sideways movement
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward with updated scales
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
