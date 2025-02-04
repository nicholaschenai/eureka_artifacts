@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0  # Reasonable forward speed limit for normalization
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Enhanced sideways penalty: Devote more weight to encourage progression in the forward direction
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    enhanced_sideways_penalty = -0.7 * sideways_velocity / max_speed

    # Temperature control to balance rewards; fine-tuned per observed performance trends
    temperature_forward = 1.7
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Improve sideways penalty scaling using temperature to enhance negative slope
    temperature_sideway = 0.6
    transformed_sideways_penalty = torch.exp(temperature_sideway * enhanced_sideways_penalty) - 1.0

    # Compute total reward, harmonizing rewards and penalties with new weights
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "enhanced_sideways_penalty": enhanced_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
