"""共享工具函数"""
import numpy as np
import os


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi]。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def heuristic_action(uav_pos, target_pos, dist_max):
    """几何启发式动作：朝目标方向的 (phi, step)。"""
    vec = target_pos - uav_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-8:
        return np.array([0.0, 0.0], dtype=np.float32)
    phi = np.arctan2(vec[1], vec[0])
    step = min(dist, dist_max)
    return np.array([phi, step], dtype=np.float32)


def mkdir(path):
    """创建目录（已存在则跳过）。"""
    os.makedirs(path, exist_ok=True)


def refine_action(
    raw_action,
    heuristic,
    prev_action,
    dist_to_target,
    dist_max,
    guidance_radius,
    near_target_radius,
    max_turn_rate,
    same_target=True,
    stagnation_steps=0,
    blend_new=0.92,
    blend_near=0.75,
    blend_guidance=0.35,
    blend_far=0.12,
    blend_angle_90=0.85,
    blend_angle_45=0.55,
    blend_stagnation=0.9,
    cruise_scale=0.75,
    cruise_min=0.65,
    prev_blend=0.65,
):
    """对策略动作做轻量后处理，让轨迹更平滑、少绕路。

    通过 blend_* 参数控制训练/测试侧不同的激进程度。
    """
    raw_phi = float(raw_action[0])
    raw_step = float(raw_action[1])
    heuristic_phi = float(heuristic[0])
    heuristic_step = float(heuristic[1])
    heading_deadband = np.deg2rad(3.0 if dist_to_target < guidance_radius else 6.0)
    anti_zigzag_band = np.deg2rad(10.0 if dist_to_target < guidance_radius else 16.0)

    if not same_target:
        blend = blend_new
    elif dist_to_target < near_target_radius:
        blend = blend_near
    elif dist_to_target < guidance_radius:
        blend = blend_guidance
    else:
        blend = blend_far

    angle_error = abs(wrap_angle(raw_phi - heuristic_phi))
    if angle_error > np.deg2rad(90):
        blend = max(blend, blend_angle_90)
    elif angle_error > np.deg2rad(45):
        blend = max(blend, blend_angle_45)
    if stagnation_steps >= 2:
        blend = max(blend, blend_stagnation)

    desired_heading_delta = wrap_angle(heuristic_phi - raw_phi)
    if abs(desired_heading_delta) < heading_deadband:
        refined_phi = heuristic_phi
    else:
        refined_phi = wrap_angle(raw_phi + blend * desired_heading_delta)
    refined_step = (1.0 - blend) * raw_step + blend * heuristic_step

    if dist_to_target > guidance_radius:
        cruise_step = min(dist_max, max(cruise_scale * heuristic_step, cruise_min * dist_max))
        refined_step = max(refined_step, cruise_step)
    elif dist_to_target < guidance_radius:
        max_step_near = min(dist_max, max(dist_to_target * 0.8, 0.03))
        refined_step = min(refined_step, max_step_near)

    if prev_action is not None and same_target:
        prev_phi = float(prev_action[0])
        prev_step = float(prev_action[1])
        prev_heading_err = wrap_angle(prev_phi - heuristic_phi)
        phi_delta = wrap_angle(refined_phi - prev_phi)
        if abs(prev_heading_err) < anti_zigzag_band and abs(phi_delta) < anti_zigzag_band:
            candidate_phi = wrap_angle(prev_phi + 0.5 * phi_delta)
            candidate_err = wrap_angle(candidate_phi - heuristic_phi)
            if prev_heading_err * candidate_err < 0:
                refined_phi = heuristic_phi
            else:
                refined_phi = candidate_phi
            phi_delta = wrap_angle(refined_phi - prev_phi)
        phi_delta = float(np.clip(phi_delta, -max_turn_rate, max_turn_rate))
        refined_phi = wrap_angle(prev_phi + phi_delta)
        heading_err_after = wrap_angle(refined_phi - heuristic_phi)
        if prev_heading_err * heading_err_after < 0 and abs(prev_heading_err) < anti_zigzag_band:
            refined_phi = heuristic_phi
        if abs(wrap_angle(refined_phi - heuristic_phi)) < heading_deadband:
            refined_phi = heuristic_phi
        refined_step = prev_blend * prev_step + (1.0 - prev_blend) * refined_step

    refined_step = float(np.clip(refined_step, 0.0, dist_max))
    return np.array([refined_phi, refined_step], dtype=np.float32)
