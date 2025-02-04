@torch.jit.script
def compute_reward(
    velocity: torch.Tensor,
    up_proj: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for forward velocity along the desired direction
    forward_velocity_reward = velocity[:, 0]  # Assuming x-direction is forward

    # Reward for staying upright and maintaining the correct projection
    upright_reward = up_proj

    # Potential-based reward to encourage movement towards target
    potential_reward = potentials - prev_potentials

    # Transformations with temperature for normalization
    velocity_temp = 1.0
    upright_temp = 0.5
    potential_temp = 0.1
    
    exp_forward_velocity_reward = torch.exp(forward_velocity_reward / velocity_temp)
    exp_upright_reward = torch.exp(upright_reward / upright_temp)
    exp_potential_reward = torch.exp(potential_reward / potential_temp)
    
    # Total reward is a weighted sum of individual components
    total_reward = exp_forward_velocity_reward + 0.1 * exp_upright_reward + 0.1 * exp_potential_reward

    # Output the total reward and individual reward components
    rewards = {
        "forward_velocity_reward": forward_velocity_reward,
        "upright_reward": upright_reward,
        "potential_reward": potential_reward
    }

    return total_reward, rewards
