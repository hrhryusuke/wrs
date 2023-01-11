import numpy as np
import math
import visualization.panda.world as wd
import basis.robot_math as rm
import modeling.geometric_model as gm
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

rbt_s = xarm.XArm7YunjiMobile(enable_cc=True)
init_jnt_values = np.loadtxt('init_jnt_values_rgt.txt')

if __name__ == '__main__':
    # rbt_s.fk(component_name="arm", jnt_values=init_jnt_values)
    # tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    # base = wd.World(cam_pos=[0, 0, 5], lookat_pos=[0, 0, tcp_pos[2]+0.1])

    # rbt_s.jaw_to(jawwidth=0)
    # rbt_s.gen_meshmodel().attach_to(base)
    # rbt_s.jaw_to(jawwidth=.042)
    # rbt_s.gen_meshmodel(rgba=[0.2, 0.2, 0.2, 0.3]).attach_to(base)
    # rbt_s.jaw_to(jawwidth=.025)
    # rbt_s.gen_meshmodel(rgba=[0.2, 0.2, 0.2, 0.3]).attach_to(base)
    # rbt_s.jaw_to(jawwidth=0)
    # rbt_s.gen_meshmodel().attach_to(base)
    # rbt_s.jaw_to(jawwidth=.085)
    # rbt_s.gen_meshmodel().attach_to(base)

    add_euler = np.array([math.radians(110), math.radians(-10), math.radians(-50)])
    add_pos = np.array([0.1, -0.08, 0.3])
    add_rot = rm.rotmat_from_euler(ai=add_euler[0], aj=add_euler[1], ak=add_euler[2])

    rbt_s.fk(component_name="arm", jnt_values=init_jnt_values)
    tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    base = wd.World(cam_pos=[3, 0, tcp_pos[2]-0.1], lookat_pos=[tcp_pos[0], 0, tcp_pos[2]-0.1])
    rbt_s.gen_meshmodel(rgba=[0, 0.5, 0.5, 0.5]).attach_to(base)
    # gm.gen_sphere(pos=tcp_pos, rgba=[1, 0, 0, 0.5]).attach_to(base)
    # gm.gen_frame(pos=tcp_pos, rotmat=tcp_rot).attach_to(base)

    added_jnt_values = rbt_s.ik(tgt_pos=tcp_pos+add_pos, tgt_rotmat=np.dot(add_rot, tcp_rot))
    # gm.gen_sphere(pos=tcp_pos).attach_to(base)
    rbt_s.fk(component_name="arm", jnt_values=added_jnt_values)
    added_tcp_pos, added_tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    rbt_s.gen_meshmodel().attach_to(base)
    # gm.gen_sphere(pos=added_tcp_pos, rgba=[1, 0, 0, 1]).attach_to(base)
    # gm.gen_frame(pos=tcp_pos+add_pos, rotmat=np.dot(add_rot, tcp_rot)).attach_to(base)
    base.run()