@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjust distance reward for finer granularity
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)

    # Increase stability of the open reward
    door_opening_deg = torch.rad2deg(cabinet_dof_pos[:, 3])
    temp_opening = 0.02
    open_reward = torch.sigmoid(temp_opening * door_opening_deg)

    # Reward component for maintaining door open posture (discarded original speed reward)
    door_open_speed = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    temp_open_speed = 0.1
    posture_reward = torch.exp(temp_open_speed * torch.abs(door_open_speed))

    # Adding a new task completion reward to give strong incentive for opening the door
    completion_threshold = 30.0  # degrees considered as "open"
    task_completion_reward = torch.where(door_opening_deg >= completion_threshold, torch.tensor(1.0, device=franka_grasp_pos.device), torch.tensor(0.0, device=franka_grasp_pos.device))

    # Combine the rewards into a total, weighted sum
    weight_distance = 0.2
    weight_open = 0.4
    weight_posture = 0.2
    weight_completion = 0.2
    total_reward = (weight_distance * distance_reward +
                    weight_open * open_reward +
                    weight_posture * posture_reward +
                    weight_completion * task_completion_reward)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "posture_reward": posture_reward,
        "task_completion_reward": task_completion_reward
    }
    
    return total_reward, reward_components
