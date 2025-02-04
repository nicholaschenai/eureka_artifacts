@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Reward for velocity improvement
    velocity_reward = potentials - prev_potentials
    
    # New component for stability, penalizing any significant angular velocity deviation
    ang_velocity_penalty = torch.sum(torch.abs(ang_velocity), dim=-1)
    stability_reward = up_vec[:, 2] - 0.1 * ang_velocity_penalty
    
    # Normalizing/re-scaling components
    velocity_temp = 0.01  # New temperature parameter for velocity
    stability_temp = 0.1  # Updated temperature parameter for stability
    
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    transformed_stability_reward = torch.exp(stability_reward * stability_temp) - 1.0
    
    # Final reward is a combined, balanced reward
    total_reward = transformed_velocity_reward + transformed_stability_reward

    # Return the components for analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
