import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr_mode = LocallyWeightedLinearRegression(tau=0.5)
    lwr_mode.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_valid, lwr_mode.predict(x_valid), 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x_train = self.x
        y_train = self.y
        tau = self.tau
        m, n = x_train.shape
        np.reshape(y_train, (-1, 1))

        m_valid, n_valid = x.shape
        y_pred = np.zeros((m_valid, 1))

        for i in range(m_valid):
            W = np.zeros((m, m))
            curX = np.reshape(x[i], (1, -1))
            for j in range(m):
                W[j][j] = 0.5 * np.exp(- np.linalg.norm((x_train-curX)[j]) ** 2 / (2 * tau ** 2))
            A = np.linalg.inv(np.dot(x_train.T, np.dot(W, x_train)))
            B = np.dot(x_train.T, W)
            theta = np.dot(A, np.dot(B, y_train))
            theta = np.reshape(theta, (-1,1))
            y_pred[i] = np.dot(curX, theta)

        return y_pred
        # *** END CODE HERE ***
