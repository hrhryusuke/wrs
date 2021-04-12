import copy
import numpy as np
import robotsim._kinematics.collisionchecker as cc


class RobotInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface'):
        # TODO self.jlcs = {}
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # collision detection
        self.cc = None
        # component map for quick access
        self.manipulator_dict = {}
        self.ft_sensor_dict = {}
        self.hnd_dict = {}

    def change_name(self, name):
        self.name = name

    def get_hnd_on_manipulator(self, manipulator_name):
        raise NotImplementedError

    def get_jnt_ranges(self, manipulator_name):
        return self.manipulator_dict[manipulator_name].get_jnt_ranges()

    def get_jnt_values(self, manipulator_name):
        return self.manipulator_dict[manipulator_name].get_jnt_values()

    def get_gl_tcp(self, manipulator_name):
        return self.manipulator_dict[manipulator_name].get_gl_tcp()

    def fix_to(self, pos, rotmat):
        return NotImplementedError

    def fk(self, manipulator_name, jnt_values):
        return NotImplementedError

    def jaw_to(self, hnd_name, jaw_width):
        self.hnd_dict[hnd_name].jaw_to(jaw_width=jaw_width)

    def ik(self,
           manipulator_name,
           tgt_pos,
           tgt_rot,
           seed_conf=None,
           tcp_jntid=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_debug=False):
        return self.manipulator_dict[manipulator_name].ik(tgt_pos,
                                                          tgt_rot,
                                                          seed_conf=seed_conf,
                                                          tcp_jntid=tcp_jntid,
                                                          tcp_loc_pos=tcp_loc_pos,
                                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                                          local_minima=local_minima,
                                                          toggle_debug=toggle_debug)

    def rand_conf(self, manipulator_name):
        return self.manipulator_dict[manipulator_name].rand_conf()

    def cvt_gl_to_loc_tcp(self, manipulator_name, gl_obj_pos, gl_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_gl_to_loc_tcp(gl_obj_pos, gl_obj_rotmat)

    def cvt_loc_tcp_to_gl(self, manipulator_name, rel_obj_pos, rel_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_loc_tcp_to_gl(rel_obj_pos, rel_obj_rotmat)

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contact_points=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :param toggle_contact_points: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20201223
        """
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             otherrobot_list=otherrobot_list,
                                             toggle_contact_points=toggle_contact_points)
        return collision_info

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='yumi_gripper_meshmodel'):
        raise NotImplementedError

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self_copy.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy
