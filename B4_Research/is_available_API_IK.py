import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import drivers.xarm.wrapper.xarm_api as xarm_api
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_sim
# import robot_con.xarm_shuidi.xarm_shuidi_x as xarm_act

base = wd.World(cam_pos=[5, -5, 2], lookat_pos=[0, 0, .5])

arm_raw_gap = np.array([0, 0, 215])
rbt_s = xarm_sim.XArmShuidi(enable_cc=True)
xarm_api = xarm_api.XArmAPI(is_radian=True, do_not_open=True)

# set init jnt values
tcp_pos, tcp_rot = rbt_s.get_gl_tcp()
rel_rot = rm.rotmat_from_euler(ai=math.radians(10), aj=math.radians(10), ak=math.radians(10))
jnt_values = rbt_s.ik(tgt_pos=tcp_pos, tgt_rotmat=rel_rot.dot(tcp_rot))
rbt_s.fk(jnt_values=jnt_values)
rbt_s.gen_meshmodel().attach_to(base)

# estimate arm pos and rot
true_tcp_pos = rbt_s.arm.jnts[-1]['gl_posq']
true_tcp_rot = rbt_s.arm.jnts[-1]['gl_rotmatq']
print(f'true info: {true_tcp_pos},\n {true_tcp_rot}')
tcp_pos, tcp_rot = rbt_s.get_gl_tcp()
arm_gap = tcp_rot.dot(arm_raw_gap)
print(f'estimated info: {(tcp_pos*1000-arm_gap)/1000},\n {tcp_rot}')

ik_result = xarm_api.get_inverse_kinematics(pose=np.hstack((tcp_pos*1000-arm_gap, rm.rotmat_to_euler(tcp_rot))))
print(ik_result)

base.run()


