@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Component 1: Encourage hand to drawer handle proximity
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = 1.0 / (1.0 + distance_to_drawer)
    distance_temperature = 1.0
    distance_reward_transformed = torch.exp(distance_temperature * distance_reward)

    # Component 2: Encourage opening the door
    door_opening = cabinet_dof_pos[..., 3]
    opening_reward = torch.clamp(door_opening, min=0.0)
    opening_temperature = 0.1
    opening_reward_transformed = torch.exp(opening_temperature * opening_reward)

    # Component 3: Penalize high velocity (jerkiness)
    velocity_penalty = torch.norm(cabinet_dof_vel[..., 3:], dim=-1)
    velocity_penalty_temperature = -0.1
    velocity_penalty_transformed = torch.exp(velocity_penalty_temperature * velocity_penalty)

    # Component 4: Successful Door Open
    successfully_open = (door_opening > 0.95).float()
    success_reward = successfully_open * 10.0

    # Total reward
    total_reward = distance_reward_transformed + opening_reward_transformed + velocity_penalty_transformed + success_reward

    # Break down of each reward component
    reward_dict = {
        "distance_reward": distance_reward,
        "opening_reward": opening_reward,
        "velocity_penalty": -velocity_penalty,  # Treat as negative component
        "success_reward": success_reward
    }

    return total_reward, reward_dict
