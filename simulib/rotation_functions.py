# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:52:21 2023

@author: Josh

@purpose: Define helper functions related to radar attitude and gimbal
  computations.
"""
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as rot
from collections import deque

# Speed of light (m/s)
c0 = 299792458.0
# Conversion from degrees to radians
DTR = np.pi / 180.0
RTD = 180.0 / np.pi
# The rotation matrix from the gimbal-pointed frame to the antenna frame (this
#   is just some swapping of axes and ends up with y of the antenna frame
#   pointing out normal of the antenna face, x pointing out the right, and
#   z pointing out the top of the antenna)
rotGPtoA = np.array([
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]])

""" Define the AESA rotation set """
rotations = {
    'AESA': {
        ('Inertial', 'Body'):
            rot.from_euler('zxy', [0, -0, -0]),  # yaw, -pitch, -roll
        ('Body', 'Antenna_Base'):
            rot.from_euler('zxy', [0, -0, -0]),  # Offset yaw, -pitch, -roll
        ('Antenna_Base', 'Antenna_Pointing'):
            rot.from_euler('zy', [-0, -0]), },  # AESA -phi, -theta
    'Gimbal': {
        ('Inertial', 'Body'):
            rot.from_euler('zxy', [0, -0, -0]),  # yaw, -pitch, -roll
        ('Body', 'Mounted_Gimbal'):
            rot.from_euler('y', -np.pi)
            * rot.from_euler('zxy', [0, -0, -0]),  # Offset yaw, -pitch, -roll
        ('Mounted_Gimbal', 'Antenna_Base'):
            rot.from_euler('zx', [0, -0]),  # Gimbal pan, -tilt
        ('Antenna_Base', 'Antenna_Pointing'):
            rot.from_euler('zy', [-0, -0]), }}  # Antenna -phi, -theta


def find_path(a_rotations, a_start, a_end):
    """
    Gets a queue with the rotation path from a_start frame to a_end frame.
      It is capable of traversing in reverse as well.
    :param a_rotations: Dictionary containing all the rotation definitions.
    :param a_start: A string with the name of the starting rotation frame.
    :param a_end: A string with the name of the ending rotation frame.
    :return: Queue with all the rotations that complete the path between the
      start and end frame.
    """
    graph = {}
    for (src, dst) in a_rotations:
        graph.setdefault(src, []).append(dst)
        # Allow reverse traversal
        graph.setdefault(dst, []).append(src)

    queue = deque([[a_start]])
    visited = set()

    while queue:
        f_path = queue.popleft()
        node = f_path[-1]

        if node == a_end:
            return f_path

        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                queue.append(f_path + [neighbor])

    return None


def compose_rotation(a_rotations, a_path):
    """
    Given the set of defined rotations, this composes the complete rotation
      through the given rotation path.
    :param a_rotations: Dictionary containing all the defined rotations.
    :param a_path: A queue with the path through the rotations.
    :return: Scipy spatial transform rotation object for the total rotation
    """

    r_total = rot.identity()

    for i in range(len(a_path) - 1):
        src = a_path[i]
        dst = a_path[i + 1]

        if (src, dst) in a_rotations:
            r = a_rotations[(src, dst)]
        elif (dst, src) in a_rotations:
            r = a_rotations[(dst, src)].inv()
        else:
            raise ValueError(f"No rotation between {src} and {dst}")

        r_total = r * r_total

    return r_total


class RotationManager(object):

    def __init__(
            self, a_beamsteer_type: str,
            a_body_to_beamsteer_z_rotation_offset_r: float = 0,
            a_body_to_beamsteer_x_rotation_offset_r: float = 0,
            a_body_to_beamsteer_y_rotation_offset_r: float = 0):
        """
        Init function for the RotationManager. The RotationManager helps with
          keeping track of rotations between frames defined for the given type
          of beamsteer.
        :param a_beamsteer_type: String with the beamsteer type
          ('Gimbal' or 'AESA')
        :param a_body_to_beamsteer_z_rotation_offset_r: The z-axis rotation
          offset to the beamsteer from the IMU body (or yaw offset) in radians
        :param a_body_to_beamsteer_x_rotation_offset_r: The x-axis rotation
          offset to the beamsteer from the IMU body (or pitch offset) in radians
        :param a_body_to_beamsteer_y_rotation_offset_r: The y-axis rotation
          offset to the beamsteer from the IMU body (or roll offset) in radians
        """
        # Store the beamsteer type and select the appropriate set of rotations
        self.beamsteer_type = a_beamsteer_type.lower()
        self.rotations = rotations[a_beamsteer_type]
        # Pre-compute some of the useful paths to avoid doing them later
        self.inertial_to_antenna_base_path = find_path(
            self.rotations, "Inertial", "Antenna_Base")
        self.inertial_to_body_path = find_path(
            self.rotations, "Body", "Inertial")
        self.pointing_to_inertial_path = find_path(
            self.rotations, "Antenna_Pointing", "Inertial")
        self.antenna_base_to_body_path = find_path(
            self.rotations, "Antenna_Base", "Body")
        # Update the offset rotation matrices
        temp_rotation = rot.from_euler(
            'zxy', [
                a_body_to_beamsteer_z_rotation_offset_r,
                -a_body_to_beamsteer_x_rotation_offset_r,
                -a_body_to_beamsteer_y_rotation_offset_r])
        # Also store the key for beamsteer rotations
        if self.beamsteer_type == 'gimbal':
            self.rotations[('Body', 'Mounted_Gimbal')] = rot.from_euler(
                'y', -np.pi) * temp_rotation
            self.beamsteer_key = ('Mounted_Gimbal', 'Antenna_Base')
            self.beamsteer_order_str = 'zx'
            self.beamsteer_p_multiplier = 1.0
        else:
            self.rotations[('Body', 'Antenna_Base')] = temp_rotation
            self.beamsteer_key = ('Antenna_Base', 'Antenna_Pointing')
            self.beamsteer_order_str = 'zy'
            self.beamsteer_p_multiplier = -1.0

        # Initialize the rotations to Null
        self.rot_inertial_to_antenna_base = compose_rotation(
            self.rotations, self.inertial_to_antenna_base_path)
        self.rot_body_to_inertial = compose_rotation(
            self.rotations, self.inertial_to_body_path)
        self.rot_antenna_pointing_to_inertial = compose_rotation(
            self.rotations, self.pointing_to_inertial_path)
        self.rot_antenna_base_to_body = compose_rotation(
            self.rotations, self.antenna_base_to_body_path)
        self.rotations_need_recomposed = True
        return

    def show_rotation_frames(self):
        """
        Prints the rotation frame paths defined for the beamsteer/radar system.
        """
        print_string = ""
        for (start, end) in self.rotations.keys():
            print_string += "\n\t" + start + " <---> " + end
        print(print_string)

    def get_rotation(
            self, a_start_frame: str, a_end_frame: str) -> Optional[rot]:
        """
        Gets the rotation from a_start_Frame to a_end_frame.
        :param a_start_frame: A string with the name of the start rotation frame
        :param a_end_frame: A string with the name of the end rotation frame
        :return: A Scipy.spatial.transform.Rotation object if the path exists,
          else it returns None
        """
        new_path = find_path(self.rotations, a_start_frame, a_end_frame)
        if new_path:
            return compose_rotation(self.rotations, new_path)
        return None

    def _compose_rotations(self):
        """
        Composes the rotations for the predefined rotation paths.
        """
        self.rot_inertial_to_antenna_base = compose_rotation(
            self.rotations, self.inertial_to_antenna_base_path)
        self.rot_body_to_inertial = compose_rotation(
            self.rotations, self.inertial_to_body_path)
        self.rot_antenna_pointing_to_inertial = compose_rotation(
            self.rotations, self.pointing_to_inertial_path)
        self.rot_antenna_base_to_body = compose_rotation(
            self.rotations, self.antenna_base_to_body_path)
        self.rotations_need_recomposed = False
        return

    def _update_ins_rotations(
            self, a_yaw_r: float, a_pitch_r: float, a_roll_r: float,
            a_recompose_rotations: bool = False):
        """
        Updates the inertial to body rotations given the attitude measurements
          from the INS.
        :param a_yaw_r: The left-handed rotation about z-axis in radians.
        :param a_pitch_r: The right-handed rotation about y-axis in radians.
        :param a_roll_r: The right-handed rotation about x-axis in radians.
        :param a_recompose_rotations: Flag to force recomposition of standard
          rotation paths. (WARNING: Only do this if explicitly calling this
          function. It is recommended that "update_rotations" function be used.)
        """
        self.rotations[('Inertial', 'Body')] = \
            rot.from_euler('zxy', [a_yaw_r, -a_pitch_r, -a_roll_r])
        self.rotations_need_recomposed = True
        if a_recompose_rotations:
            self._compose_rotations()
            self.rotations_need_recomposed = False
        return

    def _update_beamsteer_rotations(
            self, a_p_r: float, a_t_r: float,
            a_recompose_rotations: bool = False):
        """
        Updates the beamsteer specific rotations.
        :param a_p_r: Pan (for gimbal) or phi (for AESA) in radians.
        :param a_t_r: Tilt (for gimbal) or theta (for AESA) in radians.
        :param a_recompose_rotations: Flag to force recomposition of standard
          rotation paths. (WARNING: Only do this if explicitly calling this
          function. It is recommended that "update_and_recompose_all_rotations"
          function be used.)
        """
        self.rotations[self.beamsteer_key] = \
            rot.from_euler(
                self.beamsteer_order_str,
                [self.beamsteer_p_multiplier * a_p_r, -a_t_r])
        self.rotations_need_recomposed = True
        if a_recompose_rotations:
            self._compose_rotations()
            self.rotations_need_recomposed = False
        return

    def update_and_recompose_all_rotations(
            self, a_yaw_r: float, a_pitch_r: float, a_roll_r: float,
            a_p_r: float, a_t_r: float):
        """
        Updates the inertial to body rotations given the attitude measurements
          from the INS as well as the beamsteer measurements. It recomposes
          rotations for all predefine rotation paths before returning.
        :param a_yaw_r: The left-handed rotation about z-axis in radians.
        :param a_pitch_r: The right-handed rotation about y-axis in radians.
        :param a_roll_r: The right-handed rotation about x-axis in radians.
        :param a_p_r: Pan (for gimbal) or phi (for AESA) in radians.
        :param a_t_r: Tilt (for gimbal) or theta (for AESA) in radians.
        """
        # Update the INS and beamsteer rotations
        self._update_ins_rotations(a_yaw_r, a_pitch_r, a_roll_r)
        self._update_beamsteer_rotations(a_p_r, a_t_r)
        # Compose all rotations
        self._compose_rotations()
        self.rotations_need_recomposed = False
        return

    def compute_antenna_boresight_vector(self) -> np.ndarray:
        """

        :return:
        """
        if self.rotations_need_recomposed:
            print("WARNING: the rotations are potentially out of date."
                  " Please consider running _compose_rotations().")
        return np.reshape(
            self.rot_antenna_pointing_to_inertial.apply(np.array([0, 0, 1])),
            (3, 1))

    def compute_inertial_lever_arm_correction(
            self, a_pc_offset_from_beamsteer_center_ab_m: np.ndarray,
            a_beamsteer_offset_from_imu_b_m: np.ndarray) -> np.ndarray:
        """
        Computes the lever arm correction in the inertial frame given the
          offsets provided by the user in the antenna base frame and body frame.
        :param a_pc_offset_from_beamsteer_center_ab_m: 3x1 array with the phase
          center offset from the beamsteer center in the antenna base frame
          (in meters).
        :param a_beamsteer_offset_from_imu_b_m: 3x1 array with the offset from
          IMU to the beamsteer center in the body frame (in meters).
        :return: 3x1 array of the combined offset in the inertial frame.
        """
        if self.rotations_need_recomposed:
            print("WARNING: the rotations are potentially out of date."
                  " Please consider running _compose_rotations().")
        # Rotate phase center offsets from the antenna base to body frame to
        #   combine with beamsteer offset in body frame before rotating into the
        #   inertial frame and returning
        return np.reshape(
            self.rot_body_to_inertial.apply(
                self.rot_antenna_base_to_body.apply(
                    a_pc_offset_from_beamsteer_center_ab_m.flatten())
                + a_beamsteer_offset_from_imu_b_m.flatten()),
            (3, 1))

    def compute_antenna_phi_theta_from_inertial_pointing_vec(
            self, a_pointing_vec_inertial: np.ndarray) -> tuple[float, float]:
        """
        Computes the antenna phi and theta angles for a pointing vector defined
          in the inertial frame.
        :param a_pointing_vec_inertial: 3x1 array of the pointing vector in the
          inertial frame.
        :return: Phi and theta angles in radians.
        """
        if self.rotations_need_recomposed:
            print("WARNING: the rotations are potentially out of date."
                  " Please consider running _compose_rotations().")
        # Rotate the pointing vector into the antenna base frame
        pointing_vec_ab = self.rot_inertial_to_antenna_base.apply(
            a_pointing_vec_inertial.flatten())
        # Compute the phi and theta angles for the vector in antenna base frame
        phi_r = np.arctan2(pointing_vec_ab.item(1), pointing_vec_ab.item(0))
        theta_r = np.arccos(pointing_vec_ab.item(2))
        return phi_r, theta_r


def get_inertial_to_body_matrix(
        a_yaw: float, a_pitch: float, a_roll: float) -> np.ndarray:
    cy = np.cos(a_yaw)
    sy = np.sin(a_yaw)
    cp = np.cos(a_pitch)
    sp = np.sin(a_pitch)
    cr = np.cos(a_roll)
    sr = np.sin(a_roll)

    # Compute the inertial to body rotation matrix
    rot_i_to_b = np.array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp],
        [cp * sy, cp * cy, sp],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp]])
    return rot_i_to_b.swapaxes(0, 2).swapaxes(1, 2) if rot_i_to_b.ndim == 3 else rot_i_to_b


def get_body_to_gimbal_pointing_matrix(
        a_rot_b_to_mg: np.ndarray, a_pan: float, a_tilt: float) -> np.ndarray:
    cp = np.cos(a_pan)
    sp = np.sin(a_pan)
    ct = np.cos(a_tilt)
    st = np.sin(a_tilt)

    rot_mg_to_gp = np.array([
        [cp, -sp, 0 if isinstance(a_pan, float) else np.zeros_like(a_pan)],
        [sp * ct, cp * ct, st],
        [-sp * st, -cp * st, ct]])

    # Compute the body to gimbal-pointing rotation matrix
    return (rot_mg_to_gp.swapaxes(0, 2).swapaxes(1, 2) if rot_mg_to_gp.ndim == 3 else rot_mg_to_gp) @ a_rot_b_to_mg


def get_body_to_aesa_pointing_matrix(
        a_rot_b_to_a: np.ndarray, a_phi: float, a_theta: float) -> np.ndarray:
    cp = np.cos(a_phi)
    sp = np.sin(a_phi)
    ct = np.cos(a_theta)
    st = np.sin(a_theta)

    rot_a_to_ap = np.array([
        [cp * ct, sp * ct, -st],
        [-sp, cp, 0 if isinstance(a_phi, float) else np.zeros_like(a_phi)],
        [cp * st, sp * st, ct]])

    # Compute the mounted-AESA to AESA-pointing rotation matrix
    return rot_a_to_ap.swapaxes(0, 2).swapaxes(1, 2) @ a_rot_b_to_a


def get_effective_azimuth_and_elevation(bore_sight_vec):
    """Compute the effective inertial azimuth and elevation angle for a
        pointing vector"""
    eff_el_i = np.arcsin(-bore_sight_vec[2]) if bore_sight_vec.ndim == 1 else np.arcsin(-bore_sight_vec[:, 2])
    eff_az_i = np.arctan2(bore_sight_vec[0], bore_sight_vec[1]) if bore_sight_vec.ndim == 1 else np.arctan2(bore_sight_vec[:, 0], bore_sight_vec[:, 1])
    return eff_el_i, eff_az_i


def get_effective_inertial_azimuth_and_graze(bore_sight_vec):
    """Compute the effective inertial azimuth and grazing angle for the antenna
        pointing"""
    eff_graze_i = np.arcsin(-bore_sight_vec[2])
    eff_az_i = np.arctan2(bore_sight_vec[0], bore_sight_vec[1])
    return eff_graze_i, eff_az_i


def body_to_inertial(
        a_yaw: float, a_pitch: float, a_roll: float, a_xyz: np.ndarray) -> np.ndarray:
    # Compute the inertial to body rotation matrix
    rot_i_to_b = get_inertial_to_body_matrix(a_yaw, a_pitch, a_roll)
    # Multiply the transpose matrix (b-to-i) by the vector
    new_xyz = rot_i_to_b @ a_xyz if a_xyz.ndim == 1 else np.einsum('ijk,ij->ik', rot_i_to_b, a_xyz)

    return new_xyz


def inertial_to_body(
        a_yaw: float, a_pitch: float, a_roll: float, a_xyz: np.ndarray) -> np.ndarray:
    # Compute the inertial to body rotation matrix
    rot_i_to_b = get_inertial_to_body_matrix(a_yaw, a_pitch, a_roll)
    # Multiply the matrix (i-to-b) by the vector
    new_xyz = rot_i_to_b @ a_xyz

    return new_xyz


def gimbal_to_body(
        a_rot_b_to_mg: np.ndarray, a_pan: float, a_tilt: float, a_xyz: np.ndarray) -> np.ndarray:
    # Compute the body to gimbal-pointing rotation matrix
    rot_b_to_gp = get_body_to_gimbal_pointing_matrix(
        a_rot_b_to_mg, a_pan, a_tilt)
    # Multiply the transpose (gp-to-b) by the vector
    new_xyz = rot_b_to_gp.swapaxes(-1, -2) @ a_xyz
    return new_xyz


def body_to_gimbal(
        a_rot_b_to_mg: np.ndarray, a_pan: float, a_tilt: float, a_xyz: np.ndarray) -> np.ndarray:
    # Compute the body to gimbal-pointing rotation matrix
    rot_b_to_gp = get_body_to_gimbal_pointing_matrix(
        a_rot_b_to_mg, a_pan, a_tilt)
    # Multiply the matrix (b-to-gp) by the vector
    new_xyz = rot_b_to_gp @ a_xyz
    return new_xyz


def body_to_aesa(
        a_rot_b_to_a: np.ndarray, a_phi: float, a_theta: float, a_xyz: np.ndarray) -> np.ndarray:
    # Compute the body to AESA-pointing rotation matrix
    rot_b_to_ap = get_body_to_aesa_pointing_matrix(a_rot_b_to_a, a_phi, a_theta)
    # Multiply the matrix (b-to-ap) by the vector
    new_xyz = rot_b_to_ap @ a_xyz
    return new_xyz


def aesa_to_body(
        a_rot_b_to_a: np.ndarray, a_phi: float, a_theta: float, a_xyz: np.ndarray) -> np.ndarray:
    # Compute the body to AESA-pointing rotation matrix
    rot_b_to_ap = get_body_to_aesa_pointing_matrix(a_rot_b_to_a, a_phi, a_theta)
    # Multiply the transpose (ap-to-b) by the vector
    new_xyz = rot_b_to_ap.swapaxes(-1, -2) @ a_xyz
    return new_xyz


def get_aesa_phi_theta(
        a_boresight_aesa_frame: np.ndarray) -> tuple[float, float]:
    phi = np.arctan2(a_boresight_aesa_frame[1], a_boresight_aesa_frame[0]) if a_boresight_aesa_frame.ndim == 1 else np.arctan2(a_boresight_aesa_frame[:, 1], a_boresight_aesa_frame[:, 0])
    theta = np.arccos(a_boresight_aesa_frame[2]) if a_boresight_aesa_frame.ndim == 1 else np.arccos(a_boresight_aesa_frame[:, 2])
    return phi, theta


def get_aesa_pointing_from_phi_theta(
        phi: float, theta: float) -> np.ndarray:
    return np.array([-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])


def get_rotation_offset_matrix(
        a_roll0: float, a_pitch0: float, a_yaw0: float) -> np.ndarray:
    """
    Computes the body to gimbal frame rotation matrix.
    :param a_roll0: The right-handed rotation about the intermediate frame
      y-axis.
    :param a_pitch0: The right-handed rotation about the intermediate frame
      x-axis.
    :param a_yaw0: The left-handed rotation about the body frame z-axis.
    :return: 3x3 numpy array with the rotation matrix from body to gimbal.
    """
    # Pre-compute the sines and cosines for brevity
    cps0 = np.cos(a_yaw0)
    sps0 = np.sin(a_yaw0)
    cph0 = np.cos(a_roll0)
    sph0 = np.sin(a_roll0)
    cth0 = np.cos(a_pitch0)
    sth0 = np.sin(a_pitch0)

    # Explicitly compute the elements of the body-to-mounted-gimbal rotation
    #   matrix
    delta1 = cph0 * cps0 + sph0 * sth0 * sps0
    delta2 = -cph0 * sps0 + sph0 * sth0 * cps0
    delta3 = -sph0 * cth0
    delta4 = cth0 * sps0
    delta5 = cth0 * cps0
    delta6 = sth0
    delta7 = sph0 * cps0 - cph0 * sth0 * sps0
    delta8 = -sph0 * sps0 - cph0 * sth0 * cps0
    delta9 = cph0 * cth0

    r_offset = np.array([
        [-delta1, -delta2, -delta3],
        [delta4, delta5, delta6],
        [-delta7, -delta8, -delta9]])

    return r_offset


def get_aesa_rotation_offset_matrix(
        a_roll0: float, a_pitch0: float, a_yaw0: float) -> np.ndarray:
    """
    Computes the body to AESA frame rotation matrix.
    :param a_roll0: The right-handed rotation about the intermediate frame
      y-axis.
    :param a_pitch0: The right-handed rotation about the intermediate frame
      x-axis.
    :param a_yaw0: The left-handed rotation about the body frame z-axis.
    :return: 3x3 numpy array with the rotation matrix from body to AESA.
    """
    # Pre-compute the sines and cosines for brevity
    cps0 = np.cos(a_yaw0)
    sps0 = np.sin(a_yaw0)
    cph0 = np.cos(a_roll0)
    sph0 = np.sin(a_roll0)
    cth0 = np.cos(a_pitch0)
    sth0 = np.sin(a_pitch0)

    # Explicitly compute the elements of the body-to-mounted-aesa rotation
    #   matrix
    delta1 = cph0 * cps0 + sph0 * sth0 * sps0
    delta2 = -cph0 * sps0 + sph0 * sth0 * cps0
    delta3 = -sph0 * cth0
    delta4 = cth0 * sps0
    delta5 = cth0 * cps0
    delta6 = sth0
    delta7 = sph0 * cps0 - cph0 * sth0 * sps0
    delta8 = -sph0 * sps0 - cph0 * sth0 * cps0
    delta9 = cph0 * cth0

    return np.array([
        [delta1, delta2, delta3],
        [delta4, delta5, delta6],
        [delta7, delta8, delta9]])


def get_boresight_vector(
        a_r_offset: np.ndarray, a_alpha_az: float, a_alpha_el: float,
        a_yaw: float, a_pitch: float, a_roll: float) -> np.ndarray:
    """Returns a 3x1 numpy array with the normalized boresight vector in
    the inertial frame"""
    # Set the boresight pointing vector in the pointed gimbal frame
    delta_gp = np.array([0, 0, 1.0])
    # Rotate these into the body frame
    delta_b = gimbal_to_body(a_r_offset, a_alpha_az, a_alpha_el, delta_gp)
    # Finish the rotation into the inertial frame
    delta_i = body_to_inertial(a_yaw, a_pitch, a_roll, delta_b)
    # Return the boresight in the inertial frame
    return delta_i


def get_boresight_coordinate_axes(
        a_r_offset: np.ndarray, a_alpha_az: float, a_alpha_el: float,
        a_yaw: float, a_pitch: float, a_roll: float) -> np.ndarray:
    """Returns a 3x1 numpy array with the normalized boresight vector in
    the inertial frame"""
    # Set the boresight pointing vector in the pointed gimbal frame
    delta_gp = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    # Rotate these into the body frame
    delta_b = [gimbal_to_body(a_r_offset, a_alpha_az, a_alpha_el, dgp) for dgp in delta_gp]
    # Finish the rotation into the inertial frame
    delta_i = [body_to_inertial(a_yaw, a_pitch, a_roll, deb) for deb in delta_b]
    # Return the boresight coordinates
    return np.stack(delta_i, axis=-1)


def get_aesa_boresight_vector(
        a_r_offset: np.ndarray, a_phi: float, a_theta: float,
        a_yaw: float, a_pitch: float, a_roll: float) -> np.ndarray:
    # Set the boresight pointing vector in the pointed AESA frame
    delta_gp = np.array([0, 0, 1.0])
    # Rotate this into the body frame
    delta_b = aesa_to_body(
        a_r_offset, a_phi, a_theta, delta_gp)
    # Finish the rotation into the inertial frame
    delta_i = body_to_inertial(a_yaw, a_pitch, a_roll, delta_b)
    # Return the boresight in the inertial frame
    return delta_i


def rotate_ant_vec_to_inertial(
        rot_b_to_mg, pan_r, tilt_r, yaw_r, pitch_r, roll_r, pointing_vec_a):
    delta_b = rotGPtoA.T @ pointing_vec_a
    delta_c = gimbal_to_body(rot_b_to_mg, pan_r, tilt_r, delta_b)
    pointing_vec_i = body_to_inertial(yaw_r, pitch_r, roll_r, delta_c)
    # Compute the effective grazing and azimuth angle in the inertial frame
    inertial_el_r, inertial_az_r = get_effective_azimuth_and_elevation(pointing_vec_i)

    return inertial_el_r, inertial_az_r


def rotate_aesa_vec_to_inertial(
        a_rot_b_to_mbs, phi_r, theta_r, yaw_r, pitch_r, roll_r, pointing_vec_a):
    delta_b = pointing_vec_a @ a_rot_b_to_mbs
    pointing_vec_i = body_to_inertial(yaw_r, pitch_r, roll_r, delta_b)
    # Compute the effective grazing and azimuth angle in the inertial frame
    inertial_el_r, inertial_az_r = get_effective_azimuth_and_elevation(pointing_vec_i)

    return inertial_el_r, inertial_az_r


def rotate_body_to_inertial(yaw_r, pitch_r, roll_r, body_az_r, body_el_r):
    cos_el = np.cos(body_el_r)
    pointing_vec_b = np.array([
        [np.sin(body_az_r) * cos_el],
        [np.cos(body_az_r) * cos_el],
        [np.sin(body_az_r)]])

    pointing_vec_i = body_to_inertial(
        yaw_r, pitch_r, roll_r, *pointing_vec_b)
    # Compute the effective grazing and azimuth angle in the inertial frame
    inertial_el_r, inertial_az_r = get_effective_azimuth_and_elevation(
        pointing_vec_i)

    return inertial_el_r, inertial_az_r


def apply_lever_arm_corrections(
        rot_bto_mg, yaw_r, pitch_r, roll_r, pan_r, tilt_r, imu2_gimbal_offset_m,
        gimbal2_ant_offset_m, platform_pos_m):
    # We want to represent the antenna phase centers in the inertial frame. To
    #   do this is a 4-step process. 1) Rotate the gimbal2Ant offsets into
    #   the body frame via the pan and tilt info, 2) add them to the imu2Gimbal
    #   offsets, 3) rotate those results from the body frame into the
    #   inertial frame via roll, pitch, yaw, and 4) add those results to the
    #   reported IMU inertial postion

    # 1) Rotate the gimbal2Ant offsets into the body frame
    delta_b = gimbal_to_body(rot_bto_mg, pan_r, tilt_r, gimbal2_ant_offset_m)
    # 2) Add them to the imu2Gimbal offsets
    imu2_ant_in_body_m = delta_b + imu2_gimbal_offset_m
    # 3) Rotate those results from the body frame into the inertial frame
    delta_i = body_to_inertial(yaw_r, pitch_r, roll_r, imu2_ant_in_body_m)
    # 4) Add those results to the reported IMU inertial postion
    ant_pos_m = platform_pos_m + delta_i

    return ant_pos_m


def apply_aesa_lever_arm_corrections(
        a_rot_b_to_mbs, yaw_r, pitch_r, roll_r, imu_2_aesa_ref_offset_m,
        aesa_ref_2_pc_offset_m, platform_pos_m):
    # We want to represent the antenna phase centers in the inertial frame. To
    #   do this is a 4-step process. 1) Rotate the aesaref2pc offsets into
    #   the body frame via the rotation offset matrix, 2) add them to the
    #   imu2aesaref offsets, 3) rotate those results from the body frame into
    #   the inertial frame via roll, pitch, yaw, and 4) add those results to the
    #   reported IMU inertial position

    # 1) Rotate the aesaref2pc offsets into the body frame
    delta_b = a_rot_b_to_mbs.T.dot(aesa_ref_2_pc_offset_m)
    # 2) Add them to the imu2aesaref offsets
    imu_2_pc_in_body_m = delta_b + imu_2_aesa_ref_offset_m
    # 3) Rotate those results from the body frame into the inertial frame
    delta_i = body_to_inertial(
        yaw_r, pitch_r, roll_r, *imu_2_pc_in_body_m)
    # 4) Add those results to the reported IMU inertial position
    ant_pos_m = platform_pos_m + delta_i

    return ant_pos_m


if __name__ == "__main__":

    # Define the emitter position (3x1 numpy array)
    emitterPosM = np.array([[.1], [.1], [.1]])
    # Define the radar platform position (3x1 numpy array)
    platformPosM = np.array([[1.], [.1], [.1]])
    # Get the yaw, pitch, roll, pan, and tilt
    yawR = (np.random.rand() - 0.5) * 2 * np.pi
    yawR = 0
    pitchR = (np.random.rand() - 0.5) * np.pi
    pitchR = 0
    rollR = (np.random.rand() - 0.5) * np.pi
    rollR = 0
    beamsteer_is_gimbal = False
    # Let's generally call the gimbal or AESA the beamsteerer and make the
    #   commanded value P and T (for pan/tilt if gimbal or phi/theta if AESA)
    p_r = (np.random.rand() - 0.5) * 60 * DTR
    p_r = 0
    t_r = np.random.rand() * 60 * DTR
    t_r = 0
    # Get the rotation offset matrix for the body frame to mounted gimbal frame
    #   (this is based on the rotation offset angles yaw, pitch, roll listed
    #   in the XML under the gimbal settings. Should be 0, 0, and -91.5 deg.)
    if beamsteer_is_gimbal:
        # These are the rotation offsets for a gimbal from the body to the
        #   mounted beamsteer
        rollOffsetR = 0 * DTR
        pitchOffsetR = 0 * DTR
        yawOffsetR = -91.5 * DTR
        rot_b_to_mbs = get_rotation_offset_matrix(
            rollOffsetR, pitchOffsetR, yawOffsetR)
    else:
        # Override with the AESA offsets if not a gimbal and get the rotation
        #   offset matrix for the AESA from the body to the mounted beamsteer
        rollOffsetR = 179.89 * DTR
        pitchOffsetR = 61.13 * DTR
        yawOffsetR = -89.85 * DTR
        rot_b_to_mbs = get_aesa_rotation_offset_matrix(
            rollOffsetR, pitchOffsetR, yawOffsetR)

    # Define the antenna phase center separation
    phaseCenSeparationM = 19.6 / 100
    # Define the center frequency
    cenFreqHz = 9.6e9

    """ Computing the Angle of Arival in the Antenna frame for a known emitter
      location and radar platform position """
    # Compute the pointing vector for the antenna to the emitter
    emitterPointingVecI = emitterPosM - platformPosM
    # Compute range to the emitter and normalize the pointing vector
    emitterRange = \
        np.sqrt(emitterPointingVecI.T.dot(emitterPointingVecI)).item(0)
    emitterPointingVecI /= emitterRange
    # Get the effective graze and azimuth for the emitter pointing vector
    emitterEffGrazeI, emitterEffAzI = \
        get_effective_inertial_azimuth_and_graze(emitterPointingVecI)

    # Rotate pointing vector into the beamsteer frame
    deltaB = inertial_to_body(
        yawR, pitchR, rollR, emitterPointingVecI)
    if beamsteer_is_gimbal:
        deltaC = body_to_gimbal(
            rot_b_to_mbs, p_r, t_r, deltaB)
        emitterPointingVecA = rotGPtoA.dot(deltaC)
    else:
        emitterPointingVecA = body_to_aesa(
            rot_b_to_mbs, p_r, t_r, deltaB)

    # Get a beam boresight vector for either the beamsteer device
    if beamsteer_is_gimbal:
        beam_boresight_vector_I = get_boresight_vector(
            rot_b_to_mbs, p_r, t_r, yawR, pitchR, rollR)
    else:
        beam_boresight_vector_I = get_aesa_boresight_vector(
            rot_b_to_mbs, p_r, t_r, yawR, pitchR, rollR)

    # Compute angle of arrival in antenna frame and time delay
    #   emitterPointingVecA = rotItoA.dot( emitterPointingVecI )
    antEle = np.arcsin(emitterPointingVecA[2, 0])
    antAz = np.arctan2(
        emitterPointingVecA[0, 0], emitterPointingVecA[1, 0])

    # Convert the antenna azimuth angle of arrival to a time delay and
    #    a phase
    timeDelayS = np.sin(antAz) * phaseCenSeparationM / c0
    interPhaseD = 2 * np.pi * timeDelayS * cenFreqHz

    """ Computing the angle of arival in the inertial frame for a measured
      elevation and azimuth angle of arival in the gimballed-antenna frame """
    # Combine the estimated azimuth angle of arrival with a 0 degree
    #   elevation angle of arrival to convert to an inertial LOB
    # Create the antenna frame unit vector pointing in the direction
    #   of the signal
    antAzAngleOfArrivalR = (np.random.rand() - 0.5) * 3.8 * DTR
    pointingVecA = np.array([
        np.sin(antAzAngleOfArrivalR),
        0,
        np.cos(antAzAngleOfArrivalR)])

    # Rotate the antenna pointing vector to the inertial frame to get a LOB
    if beamsteer_is_gimbal:
        inertialElAngleR, inertialAzAngleR = rotate_ant_vec_to_inertial(
            rot_b_to_mbs, p_r, t_r, yawR, pitchR, rollR, pointingVecA)
    else:
        inertialElAngleR, inertialAzAngleR = rotate_aesa_vec_to_inertial(
            rot_b_to_mbs, p_r, t_r, yawR, pitchR, rollR, pointingVecA)

    """
    Apply lever-arm corrections from the IMU to the antenna's phase
      centers for a gimbal setup
    """
    # Define the IMU 2 gimbal rotation center offsets (measured in the
    #   body-frame)
    imu2GimbalOffsetM = np.array([
        [-0.6020],
        [0.1217],
        [0.8213]]).flatten()
    # Define the gimbal rotation center to antenna phase center offsets
    #   (measured in the gimbal gimbal-pointing frame) CENTER (Corresponds to
    #   Antenna_Port_3 in the XML)
    gimbal2AntCenOffsetM = np.array([
        [0.0],
        [0.0714],
        [0.0894]]).flatten()
    # LEFT (Corresponds to Antenna_Port_4 in the XML)
    gimbal2AntLeftOffsetM = np.array([
        [0.098],
        [0.0714],
        [0.0894]]).flatten()
    # RIGHT (Corresponds to Antenna_Port_5 in the XML)
    gimbal2AntRightOffsetM = np.array([
        [-0.098],
        [0.0714],
        [0.0894]]).flatten()

    # Apply the lever arm corrections for each phase center of the antenna
    # CENTER
    antCenInertialPosM = apply_lever_arm_corrections(
        rot_b_to_mbs, yawR, pitchR, rollR, p_r, t_r, imu2GimbalOffsetM,
        gimbal2AntCenOffsetM, platformPosM)
    # LEFT
    antLeftInertialPosM = apply_lever_arm_corrections(
        rot_b_to_mbs, yawR, pitchR, rollR, p_r, t_r, imu2GimbalOffsetM,
        gimbal2AntLeftOffsetM, platformPosM)
    # RIGHT
    antRightInertialPosM = apply_lever_arm_corrections(
        rot_b_to_mbs, yawR, pitchR, rollR, p_r, t_r, imu2GimbalOffsetM,
        gimbal2AntRightOffsetM, platformPosM)

    """
    Apply lever-arm corrections from the IMU to the antenna's phase
      centers for an AESA setup
    """
    # Define the IMU 2 AESA reference offset (measured in the body-frame)
    imu2AESARefOffsetM = np.array([
        [-0.6020],
        [0.1217],
        [0.8213]]).flatten()
    # Define the AESA reference to antenna phase center offsets
    #   (measured in the mounted beamsteer frame or the AESA frame in this case)
    # Right
    aesaRef2TopRightPCOffsetM = np.array([
        [0.098],
        [0.0714],
        [0.0]]).flatten()
    # RIGHT
    aesaRef2TopLeftPCOffsetM = np.array([
        [-0.098],
        [0.0714],
        [0.0]]).flatten()

    # Apply the lever arm corrections for each phase center of the antenna
    # LEFT
    antTopLeftInertialPosM = apply_lever_arm_corrections(
        rot_b_to_mbs, yawR, pitchR, rollR, p_r, t_r, imu2AESARefOffsetM,
        aesaRef2TopLeftPCOffsetM, platformPosM)
    # RIGHT
    antTopRightInertialPosM = apply_lever_arm_corrections(
        rot_b_to_mbs, yawR, pitchR, rollR, p_r, t_r, imu2AESARefOffsetM,
        aesaRef2TopRightPCOffsetM, platformPosM)



