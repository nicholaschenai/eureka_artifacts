@torch.jit.script
def compute_reward(velocity: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Running faster is rewarded
    linvel_reward_factor = 1.0
    angular_penalty_factor = 0.1
    
    # Reward based on the change in potential energy, which relates to forward movement
    progress_reward = (potentials - prev_potentials) * linvel_reward_factor
    
    # Penalize large actions to encourage smoother movements
    action_penalty = torch.sum(actions**2, dim=-1) * angular_penalty_factor
    
    # Total reward
    total_reward = progress_reward - action_penalty
    
    # Individual components of the reward can help with debugging and understanding adaptation
    reward_dict = {
        'progress_reward': progress_reward,
        'action_penalty': action_penalty
    }
    
    return total_reward, reward_dict
