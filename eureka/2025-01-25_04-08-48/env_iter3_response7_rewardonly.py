@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Adjust velocity reward with a temperature parameter
    velocity_temp = 0.3
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefine consistency reward to account for balance and angular stability
    st_temp = 5.0
    uprightness = (up_vec[:, 2] - 1.0).abs()
    angular_stability = ang_velocity.norm(p=2, dim=1)  # Magnitude of angular velocity
    combined_stability = uprightness + angular_stability
    consistency_reward = torch.exp(-combined_stability * st_temp)

    # Combine into total reward
    total_reward = 0.7 * transformed_velocity_reward + 0.3 * consistency_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "consistency_reward": consistency_reward
    }

    return total_reward, reward_dict
