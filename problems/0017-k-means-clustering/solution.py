import torch


def euclidean_distance(x, c) -> float:
    """
    Compute the Euclidean distance between two points a and b.
    a: tensor of shape (n_features,)
    b: tensor of shape (n_features,)
    Returns a float.
    """
    x = x.unsqueeze(1)  # (1, n_features)
    c = c.unsqueeze(0)  # (1, n_features)
    return torch.sqrt(torch.sum((x - c) ** 2, dim=2))


def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float) # (n,k)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float) # (c,k)
    new_centroids = centroids.clone() # (c,k)

    for i in range(max_iterations):
        
        # dist = torch.cdist(points_t, centroids) # (n,c)
        dist = euclidean_distance(points_t, new_centroids) # (n,c)
        labels = torch.argmin(dist, dim=1) # (n,)
        onehot = torch.nn.functional.one_hot(labels, num_classes=k).to(points_t.dtype) # (n,c)
        count = torch.sum(onehot, dim=0, keepdim=True).T # (c,1)
        new_centroids = (onehot.T @ points_t)/count # (c,k)/c
        
        return torch.round(new_centroids, decimals=4).numpy()


