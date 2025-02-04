@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the velocity reward
    velocity_improvement = potentials - prev_potentials
    velocity_temp = 0.15  # Adjusted temperature parameter for velocity
    transformed_velocity_reward = torch.exp(velocity_improvement * velocity_temp) - 1.0

    # Reformulate the stability reward focusing on deviation from the upright ideal vector
    stability_reform = (1.0 - up_vec[:, 2]).abs()
    stability_temp = 5.0  # Increase to allow optimization space
    transformed_stability_reward = torch.exp(-stability_reform * stability_temp)  # Stress on penalizing instability

    # Sum of both rewards with normalized scaling factors
    total_reward = 0.8 * transformed_velocity_reward + 0.2 * transformed_stability_reward

    # Dictionary of individual reward components for analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
