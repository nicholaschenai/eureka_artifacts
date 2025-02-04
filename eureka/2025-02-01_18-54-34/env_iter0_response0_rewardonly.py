@torch.jit.script
def compute_reward(velocity: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the forward velocity component as the primary reward
    forward_velocity = velocity[:, 0]  # assuming the x-axis represents the forward direction

    # Define temperature parameters for transformation functions
    velocity_temp = 1.0
    torque_temp = 0.2

    # Reward for forward velocity
    velocity_reward = torch.exp(forward_velocity / velocity_temp)

    # Penalize high torques
    torque_penalty = torch.sum(torch.abs(actions), dim=-1)
    torque_penalty = torch.exp(-torque_penalty / torque_temp)

    # Combined reward
    total_reward = velocity_reward - torque_penalty

    # Return the total reward and individual components as a dictionary
    reward_dict = {
        "velocity_reward": velocity_reward,
        "torque_penalty": torque_penalty
    }
    
    return total_reward, reward_dict
