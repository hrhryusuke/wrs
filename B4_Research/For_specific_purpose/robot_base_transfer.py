import math
import keyboard
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as rbs
from direct.task.TaskManagerGlobal import taskMgr

base = wd.World(cam_pos=[3, 1, 1.5], lookat_pos=[0, 0, 0.5])
rbt_s = rbs.XArm7YunjiMobile()
onscreen = []
pre_agv_pos_rgt = np.array([0, 0, math.radians(0)])
current_jnt_values_rgt = np.array([0, 0, math.radians(0)])

def agv_move(task):
    global onscreen, pre_agv_pos_rgt, current_jnt_values_rgt

    agv_speed_weight = .01
    agv_rotation_weight = 0.8
    arm_linear_speed = .03
    arm_angular_speed = .1

    for item in onscreen:
        item.detach()
    onscreen.clear()

    agv_pos = rbt_s.get_jnt_values("agv")
    agv_loc_rotmat = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1*agv_pos[2])
    agv_direction = np.dot(np.array([1, 0, 0]), agv_loc_rotmat)

    pressed_keys = {'w': keyboard.is_pressed('w'),
                    'a': keyboard.is_pressed('a'),
                    's': keyboard.is_pressed('s'),
                    'd': keyboard.is_pressed('d'),
                    'r': keyboard.is_pressed('r'),  # x+ global
                    't': keyboard.is_pressed('t'),  # x- global
                    'f': keyboard.is_pressed('f'),  # y+ global
                    'g': keyboard.is_pressed('g'),  # y- global
                    'v': keyboard.is_pressed('v'),  # z+ global
                    'b': keyboard.is_pressed('b'),  # z- gglobal
                    'y': keyboard.is_pressed('y'),  # r+ global
                    'u': keyboard.is_pressed('u'),  # r- global
                    'h': keyboard.is_pressed('h'),  # p+ global
                    'j': keyboard.is_pressed('j'),  # p- global
                    'n': keyboard.is_pressed('n'),  # yaw+ global
                    'm': keyboard.is_pressed('m'),  # yaw- global
                    'o': keyboard.is_pressed('o'),  # gripper open
                    'p': keyboard.is_pressed('p')}  # gripper close
    values_list = list(pressed_keys.values())
    if pressed_keys["w"] and pressed_keys["a"]:
        current_jnt_values = np.array(pre_pos + [agv_speed_weight*agv_direction[0], agv_speed_weight*agv_direction[1], math.radians(agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["w"] and pressed_keys["d"]:
        current_jnt_values = np.array(pre_pos + [agv_speed_weight * agv_direction[0], agv_speed_weight * agv_direction[1], math.radians(-agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["s"] and pressed_keys["a"]:
        current_jnt_values = np.array(pre_pos + [-agv_speed_weight * agv_direction[0], -agv_speed_weight * agv_direction[1], math.radians(-agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["s"] and pressed_keys["d"]:
        current_jnt_values = np.array(pre_pos + [-agv_speed_weight * agv_direction[0], -agv_speed_weight * agv_direction[1], math.radians(agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["w"] and sum(values_list) == 1:  # if key 'q' is pressed
        current_jnt_values = np.array(pre_pos + [agv_speed_weight * agv_direction[0], agv_speed_weight * agv_direction[1], math.radians(0)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["s"] and sum(values_list) == 1:  # if key 'q' is pressed
        current_jnt_values = np.array(pre_pos + [-agv_speed_weight * agv_direction[0], -agv_speed_weight * agv_direction[1], math.radians(0)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["a"] and sum(values_list) == 1:  # if key 'q' is pressed
        current_jnt_values = np.array(pre_pos + [0, 0, math.radians(agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["d"] and sum(values_list) == 1:  # if key 'q' is pressed
        current_jnt_values = np.array(pre_pos + [0, 0, math.radians(-agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["o"] and sum(values_list) == 1:  # if key 'q' is pressed
        rbt_s.jaw_to(jawwidth=.085)
    elif pressed_keys["p"] and sum(values_list) == 1:  # if key 'q' is pressed
        rbt_s.jaw_to(jawwidth=0)
    elif any(pressed_keys[item] for item in ['r', 't', 'f', 'g', 'v', 'b', 'y', 'u', 'h', 'j', 'n', 'm']) and\
            sum(values_list) == 1: # global
        current_jnt_values = rbt_s.get_jnt_values()
        current_arm_tcp_pos, current_arm_tcp_rotmat = rbt_s.get_gl_tcp()
        rel_pos = np.zeros(3)
        rel_rotmat = np.eye(3)
        if pressed_keys['r']:
            rel_pos = np.array([arm_linear_speed * .5, 0, 0])
        elif pressed_keys['t']:
            rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
        elif pressed_keys['f']:
            rel_pos = np.array([0, arm_linear_speed * .5, 0])
        elif pressed_keys['g']:
            rel_pos = np.array([0, -arm_linear_speed * .5, 0])
        elif pressed_keys['v']:
            rel_pos = np.array([0, 0, arm_linear_speed * .5])
        elif pressed_keys['b']:
            rel_pos = np.array([0, 0, -arm_linear_speed * .5])
        elif pressed_keys['y']:
            rel_rotmat = rm.rotmat_from_euler(arm_angular_speed*.5, 0, 0)
        elif pressed_keys['u']:
            rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed*.5, 0, 0)
        elif pressed_keys['h']:
            rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed*.5, 0)
        elif pressed_keys['j']:
            rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
        elif pressed_keys['n']:
            rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed*.5)
        elif pressed_keys['m']:
            rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed*.5)
        new_arm_tcp_pos = current_arm_tcp_pos+rel_pos
        new_arm_tcp_rotmat = rel_rotmat.dot(current_arm_tcp_rotmat)
        new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos, tgt_rotmat=new_arm_tcp_rotmat, seed_jnt_values=current_jnt_values)
        if new_jnt_values is None:
            print("Can't solve IK!!")
            return task.cont
        rbt_s.fk(jnt_values=new_jnt_values)

    onscreen.append(rbt_s.gen_meshmodel())
    onscreen[-1].attach_to(base)
    return task.cont

if __name__ == '__main__':
    gm.gen_frame(length=3, thickness=0.01).attach_to(base)
    taskMgr.doMethodLater(1/60, agv_move, "agv_move",
                          extraArgs=None,
                          appendTask=True)
    base.run()
