import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
  print("Usage: python plotter.py <title>")
  sys.exit(1)


def read_real_part(filename):

    first_numbers = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            first_number = float(line.split()[0])
            first_numbers.append(first_number)

    return first_numbers


filename1 = "x.txt"
filename2 = "x_hat.txt"
title = sys.argv[1]

x = read_real_part(filename1)
x_hat = read_real_part(filename2)

n = len(x)
t = np.linspace(0, 1, n)
plt.figure(figsize=(12, 6))
plt.plot(t[:200], x[:200], label='Original Signal', linewidth=2)
plt.plot(t[:200], x_hat[:200], label='Reconstructed Signal', linestyle='dashed', linewidth=2)
plt.legend()
plt.title(f'Original Signal vs Reconstructed Signal - {title}')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(f'{title}.png', bbox_inches='tight')
plt.show()

