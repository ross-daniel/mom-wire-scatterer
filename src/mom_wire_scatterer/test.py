import numpy as np
import constants
import numerical_integration

N = 100
L = 2.0
E0 = 1
a = 12.5e-3
f = 300e6
omega = 2 * np.pi * f
theta = np.pi / 2
beta = omega * np.sqrt(constants.eps_0 * constants.mu_0)
d = L / (N+1)
approx_degree = 100


def psi_1_neq_old(self, m, n):
    R = lambda x: np.sqrt(((m - n) * self.d - x) ** 2 + self.a ** 2)
    psi_1 = lambda x: (1 - abs(x) / self.d) * np.exp(-1j * self.beta * R(x)) / R(x)
    result = numerical_integration.numerical_integral(psi_1, self.approx_degree, (-self.d, self.d))
    # print(f'psi_1_neq m={m}, n={n}: {result}')
    return result


def psi_1_neq(m, n):
    R = lambda x: np.sqrt(((m - n) * d - x) ** 2 + a ** 2)
    psi_1 = lambda x: (1 - abs(x) / d) * np.exp(-1j * beta * R(x)) / R(x)
    result = numerical_integration.numerical_integral(psi_1, approx_degree, (-d, d))
    # print(f'psi_1_neq m={m}, n={n}: {result}')
    return result


def psi_2(m, n):
    R_0 = np.sqrt(((m - n)*d) ** 2 + a ** 2)
    R_1 = np.sqrt(((m - n - 1)*d) ** 2 + a ** 2)
    R_2 = np.sqrt(((m - n + 1)*d) ** 2 + a ** 2)
    term0 = np.exp(-1j * beta * R_0) / R_0
    term1 = np.exp(-1j * beta * R_1) / R_1
    term2 = np.exp(-1j * beta * R_2) / R_2
    result = term1 - 2 * term0 + term2
    #print(f'psi_2 m={m}, n={n}: {result}')
    return result

if __name__ == "__main__":
    Z = np.zeros((N, N), dtype=complex)
    for m in range(N):
        for n in range(N):
            if m==0 and n==1:
                print(f'magnetic potential term: {psi_1_neq(m, n)}')
                print(f'electric potential term: {psi_2(m, n)}')
            Z[m,n] = 1j * omega * (constants.mu_0 / (4 * np.pi)) * (psi_1_neq(m, n) + (1 / (beta **2 * d)) * psi_2(m, n))
    print(f'Z(0,1) = {Z[0, 1]}')
