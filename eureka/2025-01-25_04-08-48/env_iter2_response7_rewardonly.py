@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward with a focus on ensuring it remains a primary driver
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefine stability reward using angular velocity as a stability measure
    stability_temp = 20.0  # Temperature to amplify sensitivity
    stability_reward = torch.exp(-torch.norm(ang_velocity, p=2, dim=-1) * stability_temp)

    # Add a new penalty for excessive angular velocities to maintain posture stability
    ang_velocity_penalty_temp = 1.0
    ang_velocity_penalty = -torch.exp(torch.norm(ang_velocity, p=2, dim=-1) * ang_velocity_penalty_temp)

    # Aggregate all rewards
    total_reward = (
        1.0 * transformed_velocity_reward +
        0.5 * stability_reward +
        0.1 * ang_velocity_penalty
    )

    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "stability_reward": stability_reward,
        "ang_velocity_penalty": ang_velocity_penalty
    }

    return total_reward, reward_dict
