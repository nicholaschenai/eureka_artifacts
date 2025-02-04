@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 20.0  # Increase reasonable forward speed limit for normalization to encourage exploration
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Reworked sideways penalty to focus on restricting sideways motion more severely
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    enforced_sideways_penalty = -1.0 * sideways_velocity

    # Temperature control to balance rewards; fine-tuned per observed performance trends
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Improve sideways penalty scaling using temperature for clearer negative impact
    temperature_sideway = 0.9
    transformed_sideways_penalty = torch.exp(temperature_sideway * enforced_sideways_penalty) - 1.0

    # Compute total reward, harmonizing rewards and penalties with new weights
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "enforced_sideways_penalty": enforced_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
