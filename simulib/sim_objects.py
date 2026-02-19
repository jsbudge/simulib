import numpy as np
from simulation_functions import azelToVec, db
from utils import c0

class Antenna:
    def __init__(self, phase_centers: np.ndarray, fc: float = 9.6e9, array_weights: np.ndarray = None, element_pattern: object = None):
        """
        Generate an antenna array for use in simulation.
        :param phase_centers: Nx3 array of phase center locations, measured from a central point
        :param fc: Center frequency of array
        :param array_weights: Nx1 array of weights for array elements
        :param element_pattern: lambda function of phi and theta that is the antenna pattern of an individual element
        """
        self.phase_centers = phase_centers
        self.fc = fc
        self._lambda = c0 / fc
        self.n_elements = phase_centers.shape[0]
        self.k = 2 * np.pi / self._lambda
        self.array_weights = array_weights if array_weights is not None else np.ones(self.n_elements)

        # Default element pattern is a half-wave dipole oriented in z
        self.element_pattern = element_pattern if element_pattern is not None else \
            lambda phi, theta: np.cos(np.pi / 2 * np.cos(theta - np.pi / 2))**2 / np.sin(theta - np.pi / 2)**2


    def kv(self, phi, theta):
        """
        Wave vector. Direction of incoming wave with the wavenumber.
        :param phi: Azimuth angle in radians
        :param theta: Elevation angle in radians.
        :return: Wave vector.
        """
        return azelToVec(phi, theta) * self.k

    def vk(self, phi, theta):
        """
        Steering vector.
        :param phi: Azimuth angle in radians
        :param theta: Elevation angle in radians
        :return: Steering vector.
        """
        return np.exp(-1j * self.phase_centers.dot(self.kv(phi, theta)))

    def array_factor(self, phi, theta, array_weights = None):
        aw = array_weights if array_weights is not None else self.array_weights
        return abs(np.dot(aw, self.vk(phi, theta)) / self.n_elements)

    def __str__(self):
        return f'Antenna array with {self.n_elements} elements'

    def __call__(self, phi, theta):
        return self.array_factor(phi, theta) * self.element_pattern(phi, theta)

    def getAntennaPattern(self):
        phis, thetas = np.meshgrid(np.linspace(0, np.pi, 128), np.linspace(0, np.pi, 128))
        phis = phis.flatten()
        thetas = thetas.flatten()
        elements = self.element_pattern(phis, thetas)
        return db(self.array_factor(phis, thetas) * elements).reshape(128, 128)


def design_element_positions(
        a_design_freq_hz: float, a_sub_num_elements_wide: int,
        a_sub_num_elements_high: int, a_num_sub_wide: int,
        a_num_sub_high: int) -> tuple:
    f_wavelength_m = c0 / a_design_freq_hz
    # Design element width (quarter wavelength for half wavelength spacing to
    #   avoid grating lobes)
    element_width_m = f_wavelength_m / 4
    element_height_m = element_width_m
    x_spacing_m = element_width_m * 2
    y_spacing_m = x_spacing_m
    num_high = a_sub_num_elements_high * a_num_sub_high
    num_wide = a_sub_num_elements_wide * a_num_sub_wide
    antenna_height_m = num_high * y_spacing_m
    antenna_width_m = num_wide * x_spacing_m
    sub_array_height_m = a_sub_num_elements_high * y_spacing_m
    sub_array_width_m = a_sub_num_elements_wide * x_spacing_m

    # Get the positions of the elements in the full array using the antenna
    #   parameters above and assuming 0, 0 is in the center of the array
    #   with positive y-axis up, and positive x-axis to the right, and the
    #   z-axis out perpendicular from the array
    sub_elem_y_pos_m = np.arange(a_sub_num_elements_high) * y_spacing_m
    sub_elem_x_pos_m = np.arange(a_sub_num_elements_wide) * x_spacing_m
    sub_elem_x_pos_2d_m, sub_elem_y_pos_2d_m = np.meshgrid(
        sub_elem_x_pos_m, sub_elem_y_pos_m)
    elem_positions_m = np.zeros((
        a_num_sub_high, a_num_sub_wide, 3, a_sub_num_elements_high,
        a_sub_num_elements_wide))
    for row in range(a_num_sub_high):
        for col in range(a_num_sub_wide):
            # Combine them to get a 3 x num_high x num_wide array
            elem_positions_m[row, col] = \
                np.array([
                    sub_elem_x_pos_2d_m + col * sub_array_width_m
                    - antenna_width_m / 2 + element_width_m,
                    sub_elem_y_pos_2d_m + row * sub_array_height_m
                    - antenna_height_m / 2 + element_height_m,
                    np.zeros_like(sub_elem_x_pos_2d_m)])

    return (
        np.array(elem_positions_m), antenna_width_m, antenna_height_m,
        element_width_m)


def get_aesa_phi_theta(
        a_boresight_aesa_frame: np.ndarray) -> tuple[float, float]:
    phi = np.arctan2(
        a_boresight_aesa_frame.item(1), a_boresight_aesa_frame.item(0))
    theta = np.arccos(a_boresight_aesa_frame.item(2))
    return phi, theta


def get_element_phases(
        a_element_pos_m: np.ndarray, a_point_theta_r: float,
        a_point_phi_r: float,
        a_wavelength_m: float) -> tuple[np.ndarray, np.ndarray]:
    # Compute the wave number and the steering vector
    k = 2 * np.pi / a_wavelength_m
    steering_vector = np.array([
        np.sin(a_point_theta_r) * np.cos(a_point_phi_r),
        np.sin(a_point_theta_r) * np.sin(a_point_phi_r),
        np.cos(a_point_theta_r)])

    # Compute the distance travelled to each element
    element_phases_r = np.einsum(
        "nmijk,i->nmjk", a_element_pos_m, steering_vector) * k

    return element_phases_r, np.exp(-1j * element_phases_r)


def patch_pattern(
        a_obs_vectors_mbs: np.ndarray, a_patch_size_m: float,
        a_wavelength_m: float) -> np.ndarray:
    """
    Far-field radiation pattern of a square micro-strip patch (approximate
      model)

    :param a_obs_vectors_mbs: 3xNxM array of steering vectors within the
      antenna frame
    :param a_patch_size_m: Patch side length (meters).
    :param a_wavelength_m: Operating wavelength (meters).
    :return: NxM array of the magnitude of the element pattern.
    """
    k0 = 2 * np.pi / a_wavelength_m

    # Avoid division-by-zero in sinc
    def sinc(x):
        return np.sinc(x / np.pi)

    fx = sinc(0.5 * k0 * a_patch_size_m * a_obs_vectors_mbs[0, ...])
    fy = sinc(0.5 * k0 * a_patch_size_m * a_obs_vectors_mbs[1, ...])

    # cos(theta) factor for patch tilt
    ant_response = np.abs(fx * fy * a_obs_vectors_mbs[2, ...])

    return ant_response


if __name__ == '__main__':
    cen_freq_hz = 9.6e9
    wavelength = c0 / cen_freq_hz

    aesa_elem_pos_m, aesa_width_m, aesa_height_m, element_patch_size_m = \
        design_element_positions(cen_freq_hz, 5, 4, 4, 2)

    # Compute the AESA phi and theta angles for commanding it
    aesa_phi_r, aesa_theta_r = get_aesa_phi_theta(
        np.array([1, 0, 1.]) / np.sqrt(2))

    # Get the element weights for the designed AESA phi and theta
    elem_phases_r, elem_weights = get_element_phases(
        aesa_elem_pos_m, aesa_theta_r, aesa_phi_r, wavelength)

    check = Antenna(phase_centers = aesa_elem_pos_m.swapaxes(2, 4).reshape((-1, 3)), array_weights=elem_weights.flatten())
    print(check)
    kv_check = check.kv(np.pi / 4, np.pi / 2)
    vk_check = check.vk(np.pi / 4, np.pi / 2)
    af_check = check.array_factor(np.pi / 4, np.pi / 2)
    antenna_pattern = check.getAntennaPattern()
    phis, thetas = np.meshgrid(np.linspace(0, 2 * np.pi, 128), np.linspace(0, 2 * np.pi, 128))
    phis = phis.flatten()
    thetas = thetas.flatten()
    antenna_manual = check(phis, thetas)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vecs = (azelToVec(phis, thetas) * antenna_manual).T
    ax.plot_surface(vecs[:, 0].reshape((128, 128)), vecs[:, 1].reshape((128, 128)), vecs[:, 2].reshape((128, 128)), linewidth=0, antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={'projection': 'polar'},
                            layout='constrained')
    axs[0].plot(np.linspace(0, 2 * np.pi, 128), check.element_pattern(np.zeros(128), np.linspace(0, 2 * np.pi, 128)))
    axs[0].set_title('Side View (Y-Z)')
    axs[1].plot(np.linspace(0, 2 * np.pi, 128), check.element_pattern(np.linspace(0, 2 * np.pi, 128), np.zeros(128)))
    axs[1].set_title('Top View (X-Y)')

    figpat, axspat = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={'projection': 'polar'},
                            layout='constrained')
    axspat[0].plot(np.linspace(0, 2 * np.pi, 256), check(np.zeros(256), np.linspace(0, 2 * np.pi, 256)))
    axspat[0].set_title('Side View (Y-Z)')
    axspat[1].plot(np.linspace(0, 2 * np.pi, 256), check(np.linspace(0, 2 * np.pi, 256), np.zeros(256)))
    axspat[1].set_title('Top View (X-Y)')

