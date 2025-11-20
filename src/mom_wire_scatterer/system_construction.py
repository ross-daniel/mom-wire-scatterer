import numpy as np
import src.mom_wire_scatterer.constants as constants
import src.mom_wire_scatterer.numerical_integration as numerical_integration
from typing import Tuple
import matplotlib
#matplotlib.use('Agg') # or 'TkAgg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt


class MatrixConstructor:
    def __init__(self, num_divisions: int, wire_length: float, wire_radius: float, omega: float,
                 approx_degree: int = 6):
        self.N, self.M = num_divisions, num_divisions
        self.L = wire_length
        self.a = wire_radius
        self.omega = omega
        self.wavelength = 2 * np.pi * constants.c_0 / self.omega
        self.beta = 2 * np.pi / self.wavelength
        self.d = self.L / self.N
        self.approx_degree = approx_degree
        self.Z = np.zeros((self.N, self.M), dtype=np.complex128)
        for m in range(self.N):
            for n in range(self.M):
                self.Z[m, n] = self.calculate_Zmn(m, n)

    def calculate_Zmn(self, m, n):
        if m == n:
            magnetic_potential_term = self.psi_1_eq(m, n)
        else:
            magnetic_potential_term = self.psi_1_neq(m, n)
        electric_potential_term = self.psi_2(m, n)
        return (1j * self.omega * constants.mu_0 / (4 * np.pi) *
                (magnetic_potential_term +
                 (1 / (self.d * self.beta ** 2)) * electric_potential_term))

    def psi_1_eq(self, m, n):
        R = lambda x: np.sqrt(x ** 2 + self.a ** 2)
        int1 = lambda x: (1 - x / self.d) * np.cos(self.beta * R(x)) / R(x) - np.cos(self.beta * self.a) / R(x)
        int2 = lambda x: (1 - x / self.d) * np.sin(self.beta * R(x)) / R(x)
        int3 = 2 * np.cos(self.beta * self.a) * np.log((self.d + np.sqrt(self.d ** 2 + self.a ** 2)) / self.a)
        result = (2 * numerical_integration.numerical_integral(int1, self.approx_degree, (0, self.d))
                - 2j * numerical_integration.numerical_integral(int2, self.approx_degree, (0, self.d))
                + int3)
        #print(f'psi_1_eq m={m}, n={n}: {result}')
        return result

    def psi_1_neq(self, m, n):
        R = lambda x: np.sqrt(((m - n) * self.d - x) ** 2 + self.a ** 2)
        psi_1 = lambda x: (1 - abs(x) / self.d) * np.exp(-1j * self.beta * R(x)) / R(x)
        result = numerical_integration.numerical_integral(psi_1, self.approx_degree, (-self.d, self.d))
        #print(f'psi_1_neq m={m}, n={n}: {result}')
        return result

    def psi_2(self, m, n):
        R_0 = np.sqrt(((n - m)*self.d) ** 2 + self.a ** 2)
        R_1 = np.sqrt(((n - m - 1)*self.d) ** 2 + self.a ** 2)
        R_2 = np.sqrt(((n - m + 1)*self.d) ** 2 + self.a ** 2)
        term0 = np.exp(-1j * self.beta * R_0) / R_0
        term1 = np.exp(-1j * self.beta * R_1) / R_1
        term2 = np.exp(-1j * self.beta * R_2) / R_2
        result = term1 - 2 * term0 + term2
        #print(f'psi_2 m={m}, n={n}: {result}')
        return result


class ExcitationConstructor:
    def __init__(self, num_divisions: int, wire_length: float, wire_radius: float, omega: float,
                 incident_angle: float, incident_magnitude: float = 1.0):
        self.num_divisions = num_divisions
        self.wire_length = wire_length
        self.d = self.wire_length / self.num_divisions
        self.wire_radius = wire_radius
        self.omega = omega
        self.beta = 2 * np.pi * constants.c_0 / self.omega
        self.incident_magnitude = incident_magnitude
        self.theta_incident = incident_angle
        self.E_iz = np.zeros(self.num_divisions, dtype=np.complex128)
        self.find_theta_Eiz()

    def find_theta_Eiz(self):
        for division_index in range(self.num_divisions):
            z_n_prime = self.d / 2 + self.d * division_index
            # R = np.sqrt(self.source_x ** 2 + abs(self.source_z - z_n_prime) ** 2)
            R = z_n_prime * np.cos(self.theta_incident)
            self.E_iz[division_index] = self.incident_magnitude * np.exp(-1j * self.beta * R)


class WireScattererSystem:
    def __init__(self, num_divisions: int, wire_length: float, wire_radius: float, omega: float, incident_angle: float,
                 incident_magnitude: float = 1.0, approx_degree: int = 6):
        self.matrix_constructor = MatrixConstructor(num_divisions, wire_length, wire_radius, omega, approx_degree)
        self.excitation_constructor = ExcitationConstructor(num_divisions, wire_length, wire_radius, omega, incident_angle, incident_magnitude)
        self.E_iz = self.excitation_constructor.E_iz
        self.Z = self.matrix_constructor.Z
        self.In = np.linalg.solve(self.Z, self.E_iz)

    def plot_current_distribution(self, axes: list[plt.Axes]):
        assert len(axes) < 3
        z = np.linspace(0, self.matrix_constructor.L, self.matrix_constructor.N)
        axes[0].plot(z, abs(self.In))
        try:
            axes[1].plot(z, np.angle(self.In))
        except IndexError:
            print('passing only a single set of Axes to \'plot_current_distribution\' discards phase information')


if __name__ == '__main__':
    E_mag = 1.0
    L = 2
    num_divs = 100
    wire_radius = 12.5e-3
    f = 300E6
    omega = 2 * np.pi * f
    incident_angle = np.pi / 2
    scatter_system = WireScattererSystem(num_divs, L, wire_radius, omega, incident_angle)
    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()
    scatter_system.plot_current_distribution([ax1, ax2])
    ax1.set_title('Magnitude of Current Distribution along the Wire Scatterer')
    ax2.set_title('Phase of Current Distribution along the Wire Scatterer')
    plt.show()
