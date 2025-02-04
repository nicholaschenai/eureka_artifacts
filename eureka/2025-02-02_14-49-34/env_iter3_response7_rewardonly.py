@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Recalculate distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Distance reward: keep the existing form but ensure impact
    temp_distance = 1.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)

    # Reduced scale for opening reward with sharper response
    door_angle = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.05
    open_reward = torch.exp(temp_opening * door_angle)

    # Re-vamped speed reward to rewarding substantial constant speed
    temp_velocity = 0.5
    speed_reward = torch.abs(cabinet_dof_vel[:, 3]) * torch.exp(-temp_velocity * distance_to_drawer)

    # A new component: consistency reward for maintaining opening pos
    consistency_temp = 0.05
    consistency_reward = torch.tanh(consistency_temp * torch.abs(cabinet_dof_pos[:, 3]))

    # Combine the rewards into a total, weighted sum
    weight_distance = 0.2
    weight_open = 0.3
    weight_speed = 0.2
    weight_consistency = 0.3
    total_reward = (weight_distance * distance_reward + weight_open * open_reward + 
                    weight_speed * speed_reward + weight_consistency * consistency_reward)
    
    # Ensure that total reward remains non-negative and bounded
    total_reward = torch.clamp(total_reward, min=0.0, max=5.0)

    # Configure reward components into a dictionary for logging / analysis
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "consistency_reward": consistency_reward
    }

    return total_reward, reward_components
