@torch.jit.script
def compute_reward(velocity: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, ang_velocity: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward scaling
    velocity_temp = 0.1
    efficiency_temp = 0.01
    stability_temp = 0.1
    
    # Velocity in the forward direction (assuming it's the x-direction)
    velocity_reward = velocity[:, 0]  # Assuming the forward direction is along the x-axis
    velocity_reward_trans = torch.exp(velocity_reward * velocity_temp)
    
    # Progress reward based on potential change
    progress_reward = potentials - prev_potentials
    progress_reward_trans = torch.exp(progress_reward)
    
    # Stability reward to minimize angular velocity
    stability_reward = -torch.norm(ang_velocity, p=2, dim=-1)
    stability_reward_trans = torch.exp(stability_reward * stability_temp)
    
    # Efficiency reward to penalize large joint velocities
    efficiency_reward = -torch.sum(torch.abs(dof_vel), dim=-1)
    efficiency_reward_trans = torch.exp(efficiency_reward * efficiency_temp)
    
    # Total reward as weighted sum of all components
    total_reward = 0.5 * velocity_reward_trans + 0.3 * progress_reward_trans + 0.1 * stability_reward_trans + 0.1 * efficiency_reward_trans
    
    # Returning total reward and a dictionary of individual components
    reward_dict = {
        "velocity_reward": velocity_reward_trans,
        "progress_reward": progress_reward_trans,
        "stability_reward": stability_reward_trans,
        "efficiency_reward": efficiency_reward_trans
    }
    return total_reward, reward_dict
