import numpy as np
import src.mom_wire_scatterer.constants as constants
import src.mom_wire_scatterer.numerical_integration as numerical_integration
from typing import Tuple


class ZConstructor:
    def __init__(self, num_divisions: int, wire_length: float, wire_radius: float, omega: float,
                 approx_degree: int = 6):
        self.N, self.M = num_divisions, num_divisions
        self.L = wire_length
        self.a = wire_radius
        self.omega = omega
        self.beta = 2 * np.pi * constants.c_0 / self.omega
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
        return (complex(0, 1) * self.omega * constants.mu_0 *
                (magnetic_potential_term +
                 (1 / (self.d * self.beta ** 2)) * electric_potential_term))

    def psi_1_eq(self, m, n):
        R = lambda x: np.sqrt(x ** 2 + self.a ** 2)
        int1 = lambda x: (1 - x / self.d) * np.cos(self.beta * R(x)) / R(x) - np.cos(self.beta * self.a) / R(x)
        int2 = lambda x: (1 - x / self.d) * np.sin(self.beta * R(x)) / R(x)
        int3 = 2 * np.cos(self.beta * self.a) * np.log((self.d + np.sqrt(self.d ** 2 + self.a ** 2)) / self.a)
        return (2 * numerical_integration.numerical_integral(int1, self.approx_degree, (0, self.d))
                - 2j * numerical_integration.numerical_integral(int2, self.approx_degree, (0, self.d))
                + int3)

    def psi_1_neq(self, m, n):
        R = lambda x: np.sqrt(((m - n) * self.d - x) ** 2 + self.a ** 2)
        psi_1 = lambda x: (1 - abs(x) / self.d) * np.exp(-1j * self.beta * R(x)) / R(x)
        return numerical_integration.numerical_integral(psi_1, self.approx_degree, (-self.d, self.d))

    def psi_2(self, m, n):
        R_0 = np.sqrt(((n - m)*self.d) ** 2 + self.a ** 2)
        R_1 = np.sqrt(((n - m - 1)*self.d) ** 2 + self.a ** 2)
        R_2 = np.sqrt(((n - m + 1)*self.d) ** 2 + self.a ** 2)
        term1 = np.exp(-1j * self.beta * R_0) / R_0
        term2 = np.exp(-1j * self.beta * R_1) / R_1
        term3 = np.exp(-1j * self.beta * R_2) / R_2
        return term1 - 2 * term2 + term3


class ConstructExcitation:
    def __init__(self, num_divisions: int, wire_length: float, wire_radius: float, omega: float,
                 source_location: Tuple[float, float], incident_magnitude: float):
        self.num_divisions = num_divisions
        self.wire_length = wire_length
        self.d = self.wire_length / self.num_divisions
        self.wire_radius = wire_radius
        self.omega = omega
        self.beta = 2 * np.pi * constants.c_0 / self.omega
        self.source_x = source_location[0]
        self.source_z = source_location[1]
        self.incident_magnitude = incident_magnitude
        self.E_iz = np.zeros(self.num_divisions, dtype=np.complex128)
        self.theta_iz = np.zeros(self.num_divisions, dtype=np.float64)
        self.find_theta_Eiz()

    def find_theta_Eiz(self):
        for division_index in range(self.num_divisions):
            z_n_prime = self.d / 2 + self.d * division_index
            R = np.sqrt(self.source_x ** 2 + abs(self.source_z - z_n_prime) ** 2)
            self.theta_iz[division_index] = np.arctan(self.source_x / abs(self.source_z - z_n_prime))
            self.E_iz[division_index] = self.incident_magnitude * np.cos(self.theta_iz[division_index]) * np.exp(-1j * self.beta * R) / R

if __name__ == '__main__':
    E_mag = 1.0
    L = 0.5
    num_divs = 10
    wire_radius = 0.002
    f = 200E6
    omega = 2 * np.pi * f
    excitation = ConstructExcitation(num_divs, L, wire_radius, omega, (1, L/2), E_mag)
    print(excitation.E_iz)
    Z_constructor = ZConstructor(num_divs, L, wire_radius, omega)
    print(Z_constructor.Z)
    I_n_coefficients = np.linalg.solve(Z_constructor.Z, excitation.E_iz)
    print(I_n_coefficients)
