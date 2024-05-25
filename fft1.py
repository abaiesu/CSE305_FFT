import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('fft1_data.csv')

# Plot original and approximated data
plt.figure(figsize=(14, 7))
plt.plot(data['Original'], label='Original Data')
plt.plot(data['Approximated'], label='Approximated Data', linestyle='--')
plt.legend()
plt.title('Original vs Approximated Data')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.show()