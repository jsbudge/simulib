import numpy as np
from .simulation_functions import db
from .utils import c0

ELEMENT_SZ = 0.004684257


class Antenna:
    elem_positions = None
    width = None
    height = None
    elem_width = None
    phase_centers = None
    phase_center_offsets = None
    fc = None
    wavelength = None
    n_elements = None
    k = None
    element_pattern = None

    def __init__(self, elem_positions, width, height, elem_width, fc, element_pattern):
        self.elem_positions = elem_positions
        self.width = width
        self.height = height
        self.elem_width = elem_width
        self.fc = fc
        self.n_elements = np.prod(self.elem_positions.shape) // 3
        self.wavelength = c0 / fc
        self.k = 2 * np.pi / self.wavelength
        self.element_pattern = element_pattern

        el_offsets = [x for xs in [[self.elem_positions[i, j].swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 3)
                                    for j in range(self.elem_positions.shape[1])]
                                   for i in range(self.elem_positions.shape[0])] for x in xs]
        self.phase_centers = np.stack([e.mean(axis=0) for e in el_offsets], axis=0)
        self.phase_center_offsets = np.stack([e for e in el_offsets], axis=0)

    def kv(self, phi: float | np.ndarray, theta: float | np.ndarray):
        """
        Wave vector. Direction of incoming wave with the wavenumber.
        :param phi: Azimuth angle in radians
        :param theta: Elevation angle in radians.
        :return: Wave vector.
        """
        return phi_theta_vector(phi, theta) * self.k

    def vk(self, phi: float, theta: float):
        """
        Steering vector.
        :param phi: Azimuth angle in radians
        :param theta: Elevation angle in radians
        :return: Steering vector.
        """
        return np.exp(-1j * self.phase_centers.dot(self.kv(phi, theta)))

    def af(self, phi: float | np.ndarray, theta: float | np.ndarray, weights: np.ndarray):
        return np.exp(1j * np.dot(self.phase_center_offsets, self.kv(phi, theta)))[0] * weights[..., None]

    def __str__(self):
        return f'Antenna array with {self.n_elements} elements'

    def __call__(self, phi: float, theta: float):
        return self.get_weights(phi, theta) * self.element_pattern(phi, theta)

    def get_weights(self, phi: float, theta: float):
        _, elem_weights = get_element_phases(self.elem_positions, theta, phi, self.wavelength)
        return np.stack([x for xs in [[elem_weights[i, j].flatten() for j in range(elem_weights.shape[1])] for i in
                               range(elem_weights.shape[0])] for x in xs], axis=0)

    def get_full_pattern(self, phi: float, theta: float, grid_phi: int = 128, grid_theta: int = 128):
        phis, thetas = np.meshgrid(np.linspace(0., 2 * np.pi, grid_phi), np.linspace(-np.pi / 2, np.pi / 2, grid_theta))
        phis = phis.flatten()
        thetas = thetas.flatten()
        elements = self.element_pattern(phis, thetas)
        weights = self.get_weights(phi, theta)
        return db(np.sum(self.af(phis, thetas, weights) * elements[None, None, :], axis=(0, 1)))

    def get_slice(self, phi: float, theta: float, slice_phi: float = None, slice_theta: float = None, num_pts: int = 128):
        if slice_phi is not None:
            thetas = np.linspace(-np.pi / 2, np.pi / 2, num_pts)
            phis = np.zeros_like(thetas) + slice_phi
        elif slice_theta is not None:
            phis = np.linspace(0, 2 * np.pi, num_pts)
            thetas = np.zeros_like(phis) + slice_theta
        elements = self.element_pattern(phis, thetas)
        weights = self.get_weights(phi, theta)
        return abs(np.sum(self.af(phis, thetas, weights) * elements[None, None, :], axis=(0, 1)))

    def calc_beamwidth(self, phi: float, theta: float, null2null: bool = True):
        # This calculates the half beamwidth in azimuth and elevation
        weights = self.get_weights(phi, theta)
        max_gain = db(np.sum(self.af(phi, theta, weights) * self.element_pattern(phi, theta)))
        dv = -1e6
        adv = -1e6
        step = .02
        v = theta
        min_func = lambda x: max_gain - db(np.sum(self.af(0., x, weights) * self.element_pattern(0., x))) - 3
        while abs(dv) > .01:
            step = step if np.sign(adv) == np.sign(dv) else -step / (10 * (1 + abs(dv - adv)))
            v += step
            adv = dv + 0.
            dv = min_func(v)
        az_bw = abs(theta - v) * (2 if null2null else 1)

        # AGain for phi
        dv = -1e6
        adv = -1e6
        step = .02
        v = phi
        min_func = lambda x: max_gain - db(np.sum(self.af(np.pi / 2, x, weights) * self.element_pattern(np.pi / 2, x))) - 3
        while abs(dv) > .01:
            step = step if np.sign(adv) == np.sign(dv) else -step / (10 * (1 + abs(dv - adv)))
            v += step
            adv = dv + 0.
            dv = min_func(v)
        el_bw = abs(theta - v) * (2 if null2null else 1)
        return az_bw, el_bw

    def calc_max_gain(self, phi: float, theta: float):
        weights = self.get_weights(phi, theta)
        return db(np.sum(self.af(phi, theta, weights) * self.element_pattern(phi, theta)))


class AESA(Antenna):
    def __init__(self, a_design_freq_hz: float, a_sub_num_elements_wide: int, a_sub_num_elements_high: int,
                 a_num_sub_wide: int, a_num_sub_high: int, element_pattern: object = None):
        """
        Generate an antenna array for use in simulation.
        :param phase_centers: Nx3 array of phase center locations, measured from a central point
        :param fc: Center frequency of array
        :param array_weights: Nx1 array of weights for array elements
        :param element_pattern: lambda function of phi and theta that is the antenna pattern of an individual element
        """

        elem_positions, width, height, elem_width = (
            design_element_positions(a_design_freq_hz, a_sub_num_elements_wide, a_sub_num_elements_high,
                                     a_num_sub_wide, a_num_sub_high))
        wavelength = c0 / a_design_freq_hz

        # Default element pattern is a half-wave dipole oriented in z
        element_pattern = element_pattern if element_pattern is not None else \
            lambda phi, theta: abs(np.sinc(ELEMENT_SZ / wavelength * -np.cos(phi) * np.sin(theta)) *
                                   np.sinc(ELEMENT_SZ / wavelength * np.sin(phi) * np.sin(theta)) * np.cos(theta))

        super().__init__(elem_positions, width, height, elem_width, a_design_freq_hz, element_pattern)

    def get_weights(self, phi: float, theta: float):
        _, elem_weights = get_element_phases(self.elem_positions, theta, phi, self.wavelength)
        return np.stack([x for xs in [[elem_weights[i, j].flatten() for j in range(elem_weights.shape[1])] for i in
                               range(elem_weights.shape[0])] for x in xs], axis=0)

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


def phi_theta_vector(phi, theta):
    return np.array([-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])


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

    # Compute the distance traveled to each element
    element_phases_r = np.einsum(
        "nmijk,i->nmjk", a_element_pos_m, steering_vector) * k

    return element_phases_r, np.exp(-1j * element_phases_r)


if __name__ == '__main__':
    cen_freq_hz = 9.6e9
    wavelength = c0 / cen_freq_hz

    # Compute the AESA phi and theta angles for commanding it
    aesa_phi_r, aesa_theta_r = get_aesa_phi_theta(
        np.array([0., 0., 1.]))

    ant = Antenna(cen_freq_hz, 5, 4, 4, 2)
    print(ant)
    bw_az, bw_el = ant.calc_beamwidth(aesa_phi_r, aesa_theta_r)
    kv_check = ant.kv(aesa_phi_r, aesa_theta_r)
    vk_check = ant.vk(aesa_phi_r, aesa_theta_r)
    # af_check = check.array_factor(np.pi / 4, np.pi / 2)
    antenna_pattern = ant.get_full_pattern(aesa_phi_r, aesa_theta_r)
    # antenna_pattern = (antenna_pattern + 60) / (antenna_pattern.max() + 60)
    antenna_pattern[antenna_pattern < 0] = 0
    phis, thetas = np.meshgrid(np.linspace(0, 2 * np.pi, 128), np.linspace(-np.pi / 2, np.pi / 2, 128))
    phis = phis.flatten()
    thetas = thetas.flatten()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vecs = (phi_theta_vector(phis, thetas) * antenna_pattern).T
    ax.plot_surface(vecs[:, 0].reshape((128, 128)), vecs[:, 1].reshape((128, 128)), vecs[:, 2].reshape((128, 128)), linewidth=0, antialiased=False)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={'projection': 'polar'},
                            layout='constrained')
    axs[0].plot(np.linspace(-np.pi / 2, np.pi / 2, 256), ant.element_pattern(np.zeros(256) + np.pi / 2, np.linspace(-np.pi / 2, np.pi / 2, 256)))
    axs[0].set_title('Side View (Y-Z)')
    axs[1].plot(np.linspace(-np.pi / 2, np.pi / 2, 256), ant.element_pattern(np.zeros(256), np.linspace(-np.pi / 2, np.pi / 2, 256)))
    axs[1].set_title('Top View (X-Y)')

    figpat, axspat = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={'projection': 'polar'},
                            layout='constrained')
    max_gain = ant.n_elements
    axspat[0].plot(np.linspace(-np.pi / 2, np.pi / 2, 256), ant.get_slice(aesa_phi_r, aesa_theta_r, slice_phi=np.pi / 2, num_pts=256))
    axspat[0].plot(np.zeros(4) + aesa_theta_r + bw_el, np.linspace(0, max_gain, 4), c='blue')
    axspat[0].plot(np.zeros(4) + aesa_theta_r - bw_el, np.linspace(0, max_gain, 4), c='blue')
    axspat[0].set_title('Side View (Y-Z)')
    axspat[1].plot(np.linspace(-np.pi / 2, np.pi / 2, 256), ant.get_slice(aesa_phi_r, aesa_theta_r, slice_phi=0., num_pts=256))
    axspat[1].plot(np.zeros(4) + aesa_phi_r + bw_az, np.linspace(0, max_gain, 4), c='blue')
    axspat[1].plot(np.zeros(4) + aesa_phi_r - bw_az, np.linspace(0, max_gain, 4), c='blue')
    axspat[1].set_title('Top View (X-Y)')

