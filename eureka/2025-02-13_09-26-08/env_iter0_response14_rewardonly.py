@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance between robot hand's grasp and drawer's grasp position
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Calculate the reward for the drawer being opened
    drawer_opening_reward = cabinet_dof_pos[:, 3]

    # Apply exponential transformation to the distance penalty
    distance_temperature = 0.5
    distance_penalty = -torch.exp(-distance_temperature * distance_to_drawer)

    # Apply exponential transformation to the drawer opening reward
    opening_temperature = 0.1
    transformed_opening_reward = torch.exp(opening_temperature * drawer_opening_reward)

    # Calculate the total reward by combining the components
    total_reward = distance_penalty + transformed_opening_reward

    # Prepare the dictionary of individual reward components
    reward_components = {
        'distance_penalty': distance_penalty,
        'drawer_opening_reward': transformed_opening_reward
    }

    return total_reward, reward_components
