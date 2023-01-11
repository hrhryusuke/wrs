import cv2
import pickle
import struct
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xsm
import robot_con.xarm_shuidi.xarm_shuidi_x as xsc
import drivers.devices.realsense_rpi.realsense_client as realsense_client

# -----------------------------------------------------------------------------
# setting for simulator
# base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, 1])
# gm.gen_frame().attach_to(base)
rbt_s = xsm.XArmShuidi(enable_cc=True)

# setting for actual devices
rbt_x = xsc.XArmShuidiX(ip="10.2.0.201")
rc_1 = realsense_client.EtherSenseClient(address="10.2.0.202", port=18360)
# rc_2 = realsense_client.EtherSenseClient(address="10.2.0.204", port=18360)

# make window
cv2.namedWindow("xArm1_vision")
# cv2.namedWindow("xArm2_vision")
# -----------------------------------------------------------------------------

current_jnt_values = rbt_x.arm_get_jnt_values()
homeconf_jnt_values = rbt_s.get_jnt_values()
rbt_x.arm_move_jspace_path(path=[current_jnt_values, homeconf_jnt_values])
# rbt_s.gen_meshmodel().attach_to(base)

while True:
    cv2.imshow("xArm1_vision", rc_1.get_rgb_image())
    # cv2.imshow("xArm2_vision", rc_2.get_rgb_image())
    key = cv2.waitKey(1)
    if key != -1:
        break
