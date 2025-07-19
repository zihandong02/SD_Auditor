# **Updated function to generate labeled training data with biased pair generation**
def generate_labeled_data_pairwise(theta_hat, size):
    """
    Generate labeled training data for learning a classifier.
    - Bias pair generation: Higher probability when closer to theta_hat.
    - Each input consists of two concatenated points [x_1, x_2] (a 2d feature vector).
    - Output label: 1 if x_1 is closer to theta_hat, otherwise 0.
    """
    data = np.zeros((2 * size, d))
    
    for i in range(2 * size):
        if np.random.rand() < bias_probability:
            # Sample closer to theta_hat (higher probability region)
            data[i] = np.random.uniform(low=theta - 0.1 * r * s, high=theta + 0.1 * r * s, size=d)
        else:
            # Sample from the usual uniform distribution
            data[i] = np.random.uniform(low=theta_hat - r * s, high=theta_hat + r * s, size=d)
    
    distances = np.linalg.norm(data - theta, axis=1)

    # Reshape into pairs
    data_pairs = data.reshape(size, 2, d)
    distances_pairs = distances.reshape(size, 2)

    # Determine which point in each pair is closer
    closer_idx = np.argmin(distances_pairs, axis=1)  # Index of closer point

    # Concatenate features to create 2d input vectors
    X_train = np.hstack((data_pairs[:, 0, :], data_pairs[:, 1, :]))

    # Labels: 1 if first point is closer, 0 otherwise
    y_train = (closer_idx == 0).astype(int)

    return X_train, y_train

# Function to train the SVM classifier
def train_pairwise_classifier(theta_hat, size=4000):
    """
    Train an SVM classifier to predict which point in a pair is closer.
    """
    X_train, y_train = generate_labeled_data_pairwise(theta_hat, size)
    model = SVC(kernel='linear', probability=True)  # Linear SVM with probability estimates
    model.fit(X_train, y_train)  # Train the classifier
    return model

# Function to classify and select the closer data point
def classify_and_filter_pairs(model, data):
    """
    Use the trained classifier to compare pairs of data points.
    - Inputs are pairs of new generated points.
    - The model predicts which point is closer.
    - Keep only the selected points for refinement.
    """
    size = len(data) // 2  # Number of pairs
    data_pairs = data[:size * 2].reshape(size, 2, d)  # Reshape into (size, 2, d)
    
    # Prepare features for classification
    X_test = np.hstack((data_pairs[:, 0, :], data_pairs[:, 1, :]))

    # Predict which point is closer
    predictions = model.predict(X_test)

    # Select the closer data points
    selected_data = np.where(predictions[:, None], data_pairs[:, 0, :], data_pairs[:, 1, :])

    return selected_data

# Function to compare pairs and select the closer one
def pairwise_select_closer(data, theta_true):
    """
    Select the closer point from each pair based on true distances.
    
    Parameters:
    - data: Generated data points (numpy array of shape (N, d))
    - theta_vec: The true theta value (numpy array of shape (d,))
    
    Returns:
    - Selected subset of data points, keeping one from each pair
    """
    size = len(data) // 2  # Number of pairs
    data_pairs = data[:size * 2].reshape(size, 2, d)  # Reshape into (size, 2, d)

    # Compute distances of both points in each pair to theta
    dist1 = np.linalg.norm(data_pairs[:, 0, :] - theta_true, axis=1)
    dist2 = np.linalg.norm(data_pairs[:, 1, :] - theta_true, axis=1)

    # Select the closer point from each pair
    closer_idx = (dist1 < dist2).astype(int)  # 0 if first is closer, 1 if second is closer
    selected_data = data_pairs[np.arange(size), closer_idx, :]  # Choose closer points

    return selected_data