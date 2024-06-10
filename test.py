import matplotlib.pyplot as plt

def read_from_txt(filename):
    res = []
    with open(filename, 'r') as f:
        text = f.read()
    text = text.split('\n')[:-1]
    for line in text:
        l = line.split(' ')
        c = float(l[0]) # assume that the number is a real number (no complex part)
        res.append(c)
    return res

y = read_from_txt('data.txt')
y_fft = read_from_txt('fft_data.txt')

# Create a plot
plt.figure(figsize=(8, 6))

# Plotting the data
plt.plot(y, label='Data', marker='o')

# Adding titles and labels
plt.title('Sample 2D Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Adding a legend
plt.legend()


# Show the plot
plt.show()


# Create a plot
plt.figure(figsize=(8, 6))

# Plotting the data
plt.plot(y_fft, label='Data', marker='o')

# Adding titles and labels
plt.title('Sample 2D Plot FFT')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Adding a legend
plt.legend()


# Show the plot
plt.show()
