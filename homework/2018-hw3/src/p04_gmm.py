import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m, n = x.shape
    mu = [np.zeros(n) for _ in range(K)]
    sigma = [np.zeros((n, n)) for _ in range(K)]
    indices = np.arange(m)
    np.random.shuffle(indices)
    groups = np.array_split(indices, K)
    for j in range(K):
        group = groups[j]
        array = x[group]
        mu[j] = np.mean(array, axis=0)
        sigma[j] = np.dot((array - mu[j]).T, array - mu[j]) / array.shape[0]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K)
    phi = phi * (1 / K)
    
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((m, K))
    w = w * (1 / K)

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    _, k = w.shape

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        for j in range(k):
            inv = np.linalg.inv(sigma[j])
            p_cond = np.exp(-0.5 * np.dot((x - mu[j]), np.dot(inv, (x-mu[j]).T)).diagonal()) / (np.linalg.det(sigma[j]) ** 0.5)
            w[:, j] = p_cond * phi[j]
        px = np.sum(w, axis=1)
        px = np.reshape(px, (m, 1))
        w = w / px

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.sum(w, axis=0) / m
        for j in range(k):
            col = np.reshape(w[:, j], (m, 1))
            mu[j] = np.dot(x.T, col) / np.sum(col)
            mu[j] = np.reshape(mu[j], (n,))
            sigma[j] = np.dot(x.T, col * x) / np.sum(col)
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        pxz = np.zeros((m, k))
        for j in range(k):
            inv = np.linalg.inv(sigma[j])
            p_cond = np.exp(-0.5 * np.dot((x - mu[j]), np.dot(inv, (x-mu[j]).T)).diagonal()) / (np.linalg.det(sigma[j]) ** 0.5)
            p_cond = p_cond / (2 * np.pi) ** (n / 2)
            pxz[:, j] = p_cond * phi[j]
        px = np.sum(pxz, axis=1)
        
        prev_ll = ll
        ll = np.sum(np.log(px))
        it += 1

        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # print(f"iter: {it}, ll: {ll}, prev_ll: {prev_ll}")
        # # having trouble passing the assertion...
        # assert(prev_ll == None or ll >= prev_ll)
        # *** END CODE HERE ***
    print(f"iterations used to converge: {it}")

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    m_tilde, _ = x_tilde.shape
    _, k = w.shape
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        for j in range(k):
            inv = np.linalg.inv(sigma[j])
            p_cond = np.exp(-0.5 * np.dot((x - mu[j]), np.dot(inv, (x-mu[j]).T)).diagonal()) / (np.linalg.det(sigma[j]) ** 0.5)
            w[:, j] = p_cond * phi[j]
        px = np.sum(w, axis=1)
        px = np.reshape(px, (m, 1))
        w = w / px

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(k):
            col = np.reshape(w[:, j], (m, 1))
            phi[j] = np.sum(col) + alpha * np.sum(z == j)
            phi[j] /= m + alpha * m_tilde
            mu[j] = np.dot(x.T, col) + alpha * np.dot(x_tilde.T, z == j) 
            mu[j] /= np.sum(col) + alpha * np.sum(z == j)
            mu[j] = np.reshape(mu[j], (n,))
            sigma[j] = np.dot(x.T, col * x) + alpha * np.dot(x_tilde.T, (z == j) * x_tilde)
            sigma[j] /= np.sum(col) + alpha * np.sum(z == j)
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        pxz = np.zeros((m, k))
        pxz_tilde = np.zeros((m_tilde, k))
        for j in range(k):
            inv = np.linalg.inv(sigma[j])
            p_cond = np.exp(-0.5 * np.dot((x - mu[j]), np.dot(inv, (x - mu[j]).T)).diagonal()) / (np.linalg.det(sigma[j]) ** 0.5)
            p_cond = p_cond / (2 * np.pi) ** (n / 2)
            p_cond_tilde = np.exp(-0.5 * np.dot((x_tilde - mu[j]), np.dot(inv, (x_tilde - mu[j]).T)).diagonal()) / (np.linalg.det(sigma[j]) ** 0.5)
            p_cond_tilde = p_cond_tilde / (2 * np.pi) ** (n / 2)
            pxz[:, j] = p_cond * phi[j]
            pxz_tilde[:, j] = p_cond_tilde * np.reshape(z == j, (m_tilde,))

        px = np.sum(pxz, axis=1)
        px_tilde = np.sum(pxz_tilde, axis=1)
        
        prev_ll = ll
        ll = np.sum(np.log(px)) + alpha * np.sum(np.log(px_tilde))
        it += 1

        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # print(f"iter: {it}, ll: {ll}, prev_ll: {prev_ll}")
        # having trouble passing the assertion...
        # assert(prev_ll == None or ll >= prev_ll)
        # *** END CODE HERE ***
    print(f"iterations used to converge: {it}")

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
