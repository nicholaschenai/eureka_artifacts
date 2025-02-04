@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = (potentials - prev_potentials)
    
    # Reduce its impact in relation to uprightness
    velocity_temp = 0.25
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Uprightness reward with increased transformation
    uprightness_temp = 5.0
    transformed_uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)

    # Total reward combines behavior incentives fairly
    total_reward = 0.55 * transformed_velocity_reward + 0.45 * transformed_uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_uprightness_reward": transformed_uprightness_reward
    }

    return total_reward, reward_dict
