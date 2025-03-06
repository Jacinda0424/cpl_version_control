import matplotlib.pyplot as plt

# Data
concentration = [3,6,12,24,48]
unbound_fraction = [0.63,0.44,0.31,0.19,0.13]

# Plot the scatter plot
plt.scatter(concentration, unbound_fraction, color="blue", label="Data Points")

# Lables and title
plt.xlabel("Concentration")
plt.ylabel("Unbound Fraction")
plt.title("Relationship between concentration and unbound fraction")

# Show the plot
plt.show()
