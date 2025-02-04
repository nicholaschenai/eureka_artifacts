@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for moving forward; introduce a larger scale factor to amplify the effect
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Enhance sideways penalty to discourage lateral movement more effectively
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    increased_sideways_penalty = -2.0 * (sideways_velocity / max_speed)

    # Re-consider heading component as a stability reward instead
    heading_proj = torch.ones_like(forward_velocity)  # Assuming heading is aligned
    stability_reward = torch.clamp(heading_proj, min=0.0, max=1.0)  # Maintain it between 0 and 1

    # New transformed rewards with adjusted temperatures for better dynamics
    temperature_forward = 0.3
    temperature_sideways = 0.6
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * increased_sideways_penalty) - 1.0

    # Combine the components into a single total reward
    total_reward = transformed_forward_reward + transformed_sideways_penalty + 0.1 * stability_reward

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "increased_sideways_penalty": increased_sideways_penalty,
        "stability_reward": stability_reward,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
