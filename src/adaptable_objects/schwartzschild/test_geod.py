import numpy as np
import matplotlib.pyplot as plt


def t_geodesic(r, r_s):
    input_rom = r / r_s
    term2 = r_s * ((2 / 3) * input_rom ** (3 / 2)
                   + 2 * np.sqrt(input_rom)
                   + np.log(abs(np.sqrt(input_rom) - 1) / (np.sqrt(input_rom) + 1)))
    return 18 - term2


r = np.linspace(0, 10, 1000)
r_s = 2 * 1
t = t_geodesic(r, r_s)
t_tilde = t + r_s * np.log(abs(r / r_s - 1))

null = r + 2 * r_s * np.log(abs(r / r_s - 1))

plt.figure(figsize=(6, 8))
plt.plot(r, t_tilde, "b-", label="Massive geodesic")
for c in range(-10, 30, 3):
    plt.plot(r, -r + c + 0.75, "g--")
    plt.plot(r, null / 3 + c + 5, "r--")
plt.legend
plt.ylim(-1, 20)
plt.show()
