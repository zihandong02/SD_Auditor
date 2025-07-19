# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Scale parameter
r = np.sqrt(3)  # Range factor for the uniform distribution (adjust this to change the range)
ths = np.linspace(0, 0.001, 2)  # Theta values (mean)
n = 200 * 10**6  # Number of samples
d = 5  # Dimensions

# Function to compute the modified expectation for a given theta
def compute_modified_exp_uniform(th, s, r, n, d):
    # Generate random samples for X1 and X2 from uniform distribution
    x1 = np.random.uniform(low=th - r * s, high=th + r * s, size=(n, d))
    x2 = np.random.uniform(low=th - r * s, high=th + r * s, size=(n, d))
    # Compute squared norms
    norm1 = np.sum(x1**2, axis=1)
    norm2 = np.sum(x2**2, axis=1)
    # Compute the indicator and the first component of X1
    ind = (norm1 <= norm2) * x1[:, 0]
    # Compute expectation, scale by 2, and square the result
    return (2 * np.mean(ind))**2

# Compute expectations for all theta values
modified_exps = [compute_modified_exp_uniform(th, s, r, n, d) for th in ths]

# Theoretical line: theta^2
theory = 0.9* ths**2

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(ths, modified_exps, label="Monte Carlo Expectation", marker='o', linestyle='-', color='orange')
plt.plot(ths, theory, label=r"Theoretical: $\theta^2$", linestyle='--', color='blue')
plt.title("Monte Carlo vs Theoretical")
plt.xlabel("Theta ($\\theta$)")
plt.ylabel("Expectation ($\\mathbb{E}^2$)")
plt.grid()
plt.legend()
plt.show()





# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Scale parameter
r = np.sqrt(3)  # Range factor for the uniform distribution
theta = 0  # True theta (as a scalar)
m = 20  # Number of samples in the first batch
n = 10000  # Number of samples in subsequent batches
d = 10  # Dimensions
t = 200  # Number of steps
delta = 1  # Probability of selecting closer points
k = 1  # Number of repetitions

# True theta vector
theta_vec = np.ones(d) * theta

# Function to compute the mean of a dataset
def compute_mean(data):
    return np.mean(data, axis=0)

# Function to filter data points with a fixed delta probability
def probabilistic_filter_data(data1, data2, theta_vec, delta):
    # Compute distances of points in both datasets to the true theta
    dist1 = np.linalg.norm(data1 - theta_vec, axis=1)
    dist2 = np.linalg.norm(data2 - theta_vec, axis=1)
    
    # Initialize selected data
    selected_data = []
    
    # Iterate through both datasets point by point
    for i in range(len(dist1)):
        if dist1[i] < dist2[i]:
            # dist1 < dist2: Choose data2 with delta probability, otherwise choose data1
            if np.random.rand() < delta:
                selected_data.append(data1[i])
            else:
                selected_data.append(data2[i])
        else:
            # dist1 >= dist2: Choose data1 with delta probability, otherwise choose data2
            if np.random.rand() < delta:
                selected_data.append(data2[i])
            else:
                selected_data.append(data1[i])
    
    # Convert the list to a NumPy array and return
    return np.array(selected_data)

# Perform the process k times
all_distances = []  # To store distances for all repetitions
for _ in range(k):
    distances = []  # Store distances for a single repetition
    mean = None  # Initialize the mean
    for step in range(t):
        if step == 0:
            # First step: Generate m samples uniformly around true theta
            data = np.random.uniform(low=theta_vec - r * s, high=theta_vec + r * s, size=(m, d))
            mean = compute_mean(data)
        else:
            # Subsequent steps: Generate n samples around the estimated mean from the previous step
            data1 = np.random.uniform(low=mean - r * s, high=mean + r * s, size=(n, d))
            data2 = np.random.uniform(low=mean - r * s, high=mean + r * s, size=(n, d))
            # Probabilistically filter data based on proximity to the true theta
            filtered_data = probabilistic_filter_data(data1, data2, theta_vec, delta)
            # Compute the mean of the filtered data
            mean = compute_mean(filtered_data)
        
        # Compute distance to true theta
        distance_to_theta = np.linalg.norm(mean - theta_vec)**2
        distances.append(distance_to_theta)
    all_distances.append(distances)

# Compute the average error over all repetitions
average_distances = np.mean(all_distances, axis=0)

# Plot the average error over steps
plt.figure(figsize=(8, 6))
plt.plot(range(t), average_distances, marker='o', linestyle='-', color='blue', label="Average Distance to True Theta")
plt.title(f"Average Distance to True Theta Over {k} Repetitions")
plt.xlabel("Step")
plt.ylabel("Average Distance")
plt.grid()
plt.legend()
plt.show()

# Print the average distances for inspection
for step, avg_distance in enumerate(average_distances):
    print(f"Step {step}: Average Distance to True Theta = {avg_distance:.6f}")






# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Standard deviation for Gaussian distribution
theta = 0  # True theta (as a scalar)
m = 20  # Number of samples in the first batch
n = 10000  # Number of samples in subsequent batches
d = 5  # Dimensions
t = 20  # Number of steps
delta = 1  # Probability of selecting closer points
k = 1  # Number of repetitions

# True theta vector
theta_vec = np.ones(d) * theta

# Function to compute the mean of a dataset
def compute_mean(data):
    return np.mean(data, axis=0)

# Function to filter data points with a delta probability of selecting closer points
def probabilistic_filter_data(data1, data2, theta_vec, delta):
    # Compute distances of points in both datasets to the true theta
    dist1 = np.linalg.norm(data1 - theta_vec, axis=1)
    dist2 = np.linalg.norm(data2 - theta_vec, axis=1)
    
    # Determine probabilistic selection
    prob1 = delta * (dist1 < dist2.mean()) + (1 - delta) * (dist1 >= dist2.mean())
    prob2 = delta * (dist2 < dist1.mean()) + (1 - delta) * (dist2 >= dist1.mean())
    
    # Generate random probabilities for selection
    random_probs1 = np.random.rand(len(dist1))
    random_probs2 = np.random.rand(len(dist2))
    
    # Select points based on the generated probabilities
    selected_data1 = data1[random_probs1 < prob1]
    selected_data2 = data2[random_probs2 < prob2]
    
    # Combine the selected datasets
    return np.vstack((selected_data1, selected_data2))

# Perform the process k times
all_distances = []  # To store distances for all repetitions
for _ in range(k):
    distances = []  # Store distances for a single repetition
    mean = None  # Initialize the mean
    for step in range(t):
        if step == 0:
            # First step: Generate m samples from Gaussian distribution centered at true theta
            data = np.random.normal(loc=theta_vec, scale=s, size=(m, d))
            mean = compute_mean(data)
        else:
            # Subsequent steps: Generate n samples around the estimated mean from the previous step
            data1 = np.random.normal(loc=mean, scale=s, size=(n, d))
            data2 = np.random.normal(loc=mean, scale=s, size=(n, d))
            # Probabilistically filter data based on proximity to the true theta
            filtered_data = probabilistic_filter_data(data1, data2, theta_vec, delta)
            # Compute the mean of the filtered data
            mean = compute_mean(filtered_data)
        
        # Compute distance to true theta
        distance_to_theta = np.linalg.norm(mean - theta_vec)**2
        distances.append(distance_to_theta)
    all_distances.append(distances)

# Compute the average error over all repetitions
average_distances = np.mean(all_distances, axis=0)

# Plot the average error over steps
plt.figure(figsize=(8, 6))
plt.plot(range(t), average_distances, marker='o', linestyle='-', color='blue', label="Average Distance to True Theta")
plt.title(f"Average Distance to True Theta Over {k} Repetitions (Gaussian)")
plt.xlabel("Step")
plt.ylabel("Average Distance")
plt.grid()
plt.legend()
plt.show()

# Print the average distances for inspection
for step, avg_distance in enumerate(average_distances):
    print(f"Step {step}: Average Distance to True Theta = {avg_distance:.6f}")





# %%
# Adaptive delta: Adjusts the probability of selecting closer points based on distance differences
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Scale parameter for uniform distribution
theta = 0  # True theta (as a scalar)
m = 20  # Number of samples in the first batch
n = 10**5  # Number of samples in subsequent batches
d = 5  # Dimensions
t = 10  # Number of steps
delta_default = 0.8  # Default probability of selecting closer points
rho = 1.3  # Threshold for distance difference
k = 6  # Number of repetitions

# True theta vector
theta_vec = np.ones(d) * theta

# Function to compute the mean of a dataset
def compute_mean(data):
    return np.mean(data, axis=0)

# Function to filter data points with an adjusted delta probability
def probabilistic_filter_data(data1, data2, theta_vec, delta_default, rho):
    # Compute distances of points in both datasets to the true theta
    dist1 = np.linalg.norm(data1 - theta_vec, axis=1)
    dist2 = np.linalg.norm(data2 - theta_vec, axis=1)
    
    # Initialize selected data
    selected_data = []
    
    # Iterate through both datasets point by point
    for i in range(len(dist1)):
        # Adjust delta dynamically based on distance difference
        delta = delta_default if abs(dist1[i] - dist2[i]) >= rho else 0.5
        
        if dist1[i] < dist2[i]:
            # dist1 < dist2: Choose data1 with delta probability, otherwise choose data2
            if np.random.rand() < delta:
                selected_data.append(data1[i])
            else:
                selected_data.append(data2[i])
        else:
            # dist1 >= dist2: Choose data2 with delta probability, otherwise choose data1
            if np.random.rand() < delta:
                selected_data.append(data2[i])
            else:
                selected_data.append(data1[i])
    
    # Convert the list to a NumPy array and return
    return np.array(selected_data)

# Perform the process k times
all_distances = []  # To store distances for all repetitions
for _ in range(k):
    distances = []  # Store distances for a single repetition
    mean = None  # Initialize the mean
    for step in range(t):
        if step == 0:
            # First step: Generate m samples uniformly around true theta
            data = np.random.uniform(low=theta_vec - s, high=theta_vec + s, size=(m, d))
            mean = compute_mean(data)
        else:
            # Subsequent steps: Generate n samples around the estimated mean from the previous step
            data1 = np.random.uniform(low=mean - s, high=mean + s, size=(n, d))
            data2 = np.random.uniform(low=mean - s, high=mean + s, size=(n, d))
            # Probabilistically filter data based on proximity to the true theta
            filtered_data = probabilistic_filter_data(data1, data2, theta_vec, delta_default, rho)
            # Compute the mean of the filtered data
            mean = compute_mean(filtered_data)
        
        # Compute distance to true theta
        distance_to_theta = np.linalg.norm(mean - theta_vec)**2
        distances.append(distance_to_theta)
    all_distances.append(distances)

# Compute the average error over all repetitions
average_distances = np.mean(all_distances, axis=0)

# Plot the average error over steps
plt.figure(figsize=(8, 6))
plt.plot(range(t), average_distances, marker='o', linestyle='-', color='blue', label="Average Distance to True Theta")
plt.title(f"Average Distance to True Theta Over {k} Repetitions (Uniform, Adaptive Delta)")
plt.xlabel("Step")
plt.ylabel("Average Distance")
plt.grid()
plt.legend()
plt.show()

# Print the average distances for inspection
for step, avg_distance in enumerate(average_distances):
    print(f"Step {step}: Average Distance to True Theta = {avg_distance:.6f}")






# %%
# Adaptive delta: Adjusts the probability of selecting closer points based on distance differences
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Standard deviation for Gaussian distribution
theta = 0  # True theta (as a scalar)
m = 20  # Number of samples in the first batch
n = 10**4  # Number of samples in subsequent batches
d = 5  # Dimensions
t = 10  # Number of steps
delta_default = 1  # Default probability of selecting closer points
rho = 1  # Threshold for distance difference
k = 10  # Number of repetitions

# True theta vector
theta_vec = np.ones(d) * theta

# Function to compute the mean of a dataset
def compute_mean(data):
    return np.mean(data, axis=0)

# Function to filter data points with an adjusted delta probability
def probabilistic_filter_data(data1, data2, theta_vec, delta_default, rho):
    # Compute distances of points in both datasets to the true theta
    dist1 = np.linalg.norm(data1 - theta_vec, axis=1)
    dist2 = np.linalg.norm(data2 - theta_vec, axis=1)
    
    # Initialize selected data
    selected_data = []
    
    # Iterate through both datasets point by point
    for i in range(len(dist1)):
        # Adjust delta dynamically based on distance difference
        delta = delta_default if abs(dist1[i] - dist2[i]) >= rho else 0.5
        
        if dist1[i] < dist2[i]:
            # dist1 < dist2: Choose data1 with delta probability, otherwise choose data2
            if np.random.rand() < delta:
                selected_data.append(data1[i])
            else:
                selected_data.append(data2[i])
        else:
            # dist1 >= dist2: Choose data2 with delta probability, otherwise choose data1
            if np.random.rand() < delta:
                selected_data.append(data2[i])
            else:
                selected_data.append(data1[i])
    
    # Convert the list to a NumPy array and return
    return np.array(selected_data)

# Perform the process k times
all_distances = []  # To store distances for all repetitions
for _ in range(k):
    distances = []  # Store distances for a single repetition
    mean = None  # Initialize the mean
    for step in range(t):
        if step == 0:
            # First step: Generate m samples from Gaussian distribution centered at true theta
            data = np.random.normal(loc=theta_vec, scale=s, size=(m, d))
            mean = compute_mean(data)
        else:
            # Subsequent steps: Generate n samples around the estimated mean from the previous step
            data1 = np.random.normal(loc=mean, scale=s, size=(n, d))
            data2 = np.random.normal(loc=mean, scale=s, size=(n, d))
            # Probabilistically filter data based on proximity to the true theta
            filtered_data = probabilistic_filter_data(data1, data2, theta_vec, delta_default, rho)
            # Compute the mean of the filtered data
            mean = compute_mean(filtered_data)
        
        # Compute distance to true theta
        distance_to_theta = np.linalg.norm(mean - theta_vec)**2
        distances.append(distance_to_theta)
    all_distances.append(distances)

# Compute the average error over all repetitions
average_distances = np.mean(all_distances, axis=0)

# Plot the average error over steps
plt.figure(figsize=(8, 6))
plt.plot(range(t), average_distances, marker='o', linestyle='-', color='blue', label="Average Distance to True Theta")
plt.title(f"Average Distance to True Theta Over {k} Repetitions (Gaussian, Adaptive Delta)")
plt.xlabel("Step")
plt.ylabel("Average Distance")
plt.grid()
plt.legend()
plt.show()

# Print the average distances for inspection
for step, avg_distance in enumerate(average_distances):
    print(f"Step {step}: Average Distance to True Theta = {avg_distance:.6f}")






# %%
# Adaptive delta with exponential function and pointwise probabilistic filtering
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Standard deviation for Gaussian distribution
theta = 0.001  # True theta (as a scalar)
m = 20  # Number of samples in the first batch
n = 10**4  # Number of samples in subsequent batches
d = 5  # Dimensions
t = 20  # Number of steps
tau = 10  # Parameter for exponential weighting
k = 1  # Number of repetitions

# True theta vector
theta_vec = np.ones(d) * theta

# Function to compute the mean of a dataset
def compute_mean(data):
    return np.mean(data, axis=0)

# Function to filter data points with dynamic delta and pointwise comparison
def probabilistic_filter_data(data1, data2, theta_vec, tau):
    # Compute distances of points in both datasets to the true theta
    dist1 = np.linalg.norm(data1 - theta_vec, axis=1)
    dist2 = np.linalg.norm(data2 - theta_vec, axis=1)
    
    # Compute dynamic delta for each point
    delta = np.exp(dist2 / tau) / (np.exp(dist1 / tau) + np.exp(dist2 / tau))
    
    # Initialize selected data
    selected_data = []
    
    # Iterate through both datasets point by point
    for i in range(len(dist1)):
        if np.random.rand() < delta[i]:
            selected_data.append(data1[i])
        else:
            selected_data.append(data2[i])
    
    # Convert the list to a NumPy array and return
    return np.array(selected_data)

# Perform the process k times
all_distances = []  # To store distances for all repetitions
for _ in range(k):
    distances = []  # Store distances for a single repetition
    mean = None  # Initialize the mean
    for step in range(t):
        if step == 0:
            # First step: Generate m samples from Gaussian distribution centered at true theta
            data = np.random.normal(loc=theta_vec, scale=s, size=(m, d))
            mean = compute_mean(data)
        else:
            # Subsequent steps: Generate n samples around the estimated mean from the previous step
            data1 = np.random.normal(loc=mean, scale=s, size=(n, d))
            data2 = np.random.normal(loc=mean, scale=s, size=(n, d))
            # Probabilistically filter data based on proximity to the true theta
            filtered_data = probabilistic_filter_data(data1, data2, theta_vec, tau)
            # Compute the mean of the filtered data
            mean = compute_mean(filtered_data)
        
        # Compute distance to true theta
        distance_to_theta = np.linalg.norm(mean - theta_vec)**2
        distances.append(distance_to_theta)
    all_distances.append(distances)

# Compute the average error over all repetitions
average_distances = np.mean(all_distances, axis=0)

# Plot the average error over steps
plt.figure(figsize=(8, 6))
plt.plot(range(t), average_distances, marker='o', linestyle='-', color='blue', label="Average Distance to True Theta")
plt.title(f"Average Distance to True Theta Over {k} Repetitions (Gaussian, Dynamic Delta)")
plt.xlabel("Step")
plt.ylabel("Average Distance")
plt.grid()
plt.legend()
plt.show()

# Print the average distances for inspection
for step, avg_distance in enumerate(average_distances):
    print(f"Step {step}: Average Distance to True Theta = {avg_distance:.6f}")






# %%
# Adaptive delta with exponential function and pointwise probabilistic filtering (Uniform Distribution)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 1  # Range parameter for uniform distribution
theta = 0.001  # True theta (as a scalar)
m = 20  # Number of samples in the first batch
n = 10**4  # Number of samples in subsequent batches
d = 5  # Dimensions
t = 200  # Number of steps
tau = 5  # Parameter for exponential weighting
k = 10  # Number of repetitions

# True theta vector
theta_vec = np.ones(d) * theta

# Function to compute the mean of a dataset
def compute_mean(data):
    return np.mean(data, axis=0)

# Function to filter data points with dynamic delta and pointwise comparison
def probabilistic_filter_data(data1, data2, theta_vec, tau):
    # Compute distances of points in both datasets to the true theta
    dist1 = np.linalg.norm(data1 - theta_vec, axis=1)
    dist2 = np.linalg.norm(data2 - theta_vec, axis=1)
    
    # Compute dynamic delta for each point
    delta = np.exp(dist2 / tau) / (np.exp(dist1 / tau) + np.exp(dist2 / tau))
    
    # Initialize selected data
    selected_data = []
    
    # Iterate through both datasets point by point
    for i in range(len(dist1)):
        if np.random.rand() < delta[i]:
            selected_data.append(data1[i])
        else:
            selected_data.append(data2[i])
    
    # Convert the list to a NumPy array and return
    return np.array(selected_data)

# Perform the process k times
all_distances = []  # To store distances for all repetitions
for _ in range(k):
    distances = []  # Store distances for a single repetition
    mean = None  # Initialize the mean
    for step in range(t):
        if step == 0:
            # First step: Generate m samples uniformly around true theta
            data = np.random.uniform(low=theta_vec - s, high=theta_vec + s, size=(m, d))
            mean = compute_mean(data)
        else:
            # Subsequent steps: Generate n samples uniformly around the estimated mean
            data1 = np.random.uniform(low=mean - s, high=mean + s, size=(n, d))
            data2 = np.random.uniform(low=mean - s, high=mean + s, size=(n, d))
            # Probabilistically filter data based on proximity to the true theta
            filtered_data = probabilistic_filter_data(data1, data2, theta_vec, tau)
            # Compute the mean of the filtered data
            mean = compute_mean(filtered_data)
        
        # Compute distance to true theta
        distance_to_theta = np.linalg.norm(mean - theta_vec)**2
        distances.append(distance_to_theta)
    all_distances.append(distances)

# Compute the average error over all repetitions
average_distances = np.mean(all_distances, axis=0)

# Plot the average error over steps
plt.figure(figsize=(8, 6))
plt.plot(range(t), average_distances, marker='o', linestyle='-', color='blue', label="Average Distance to True Theta")
plt.title(f"Average Distance to True Theta Over {k} Repetitions (Uniform, Dynamic Delta)")
plt.xlabel("Step")
plt.ylabel("Average Distance")
plt.grid()
plt.legend()
plt.show()

# Print the average distances for inspection
for step, avg_distance in enumerate(average_distances):
    print(f"Step {step}: Average Distance to True Theta = {avg_distance:.6f}")
# %%
