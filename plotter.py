import matplotlib.pyplot as plt
import re
import numpy as np


def read_array_from_txt(filename):

    out = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            out.append(float(line))

    return out

filename = "x.txt"
x = read_array_from_txt(filename)

filename = "x_hat.txt"
x_hat = read_array_from_txt(filename)

n = len(x)
t = np.linspace(0, 1, n)
plt.figure(figsize=(12, 6))
plt.plot(t[:150], x[:150], label='Original Signal', linewidth=2)
plt.plot(t[:150], x_hat[:150], label='Reconstructed Signal', linestyle='dashed', linewidth=2)
plt.legend()
plt.title('Original Signal vs Reconstructed Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig('test.png', bbox_inches='tight')
plt.show()

