@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Enhance sideways penalty to make errors more impactful
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    magnitude_factor = 2.0
    enlarged_sideways_penalty = -magnitude_factor * sideways_velocity / max_speed

    # Adjust temperatures and transformations for balance and efficiency gains
    temperature_forward = 1.0
    exp_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    temperature_sideways = 1.0
    exp_sideways_penalty = torch.exp(temperature_sideways * enlarged_sideways_penalty) - 1.0

    # Compute the total reward correctly reflecting the objectives of the task
    total_reward = exp_forward_reward + exp_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "enlarged_sideways_penalty": enlarged_sideways_penalty,
        "exp_forward_reward": exp_forward_reward,
        "exp_sideways_penalty": exp_sideways_penalty
    }
