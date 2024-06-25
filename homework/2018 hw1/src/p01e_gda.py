import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    theta = clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%.1f')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters. Shape (n+1, 1)
        """
        # *** START CODE HERE ***
        m,n = x.shape
        np.reshape(y, (m, 1))
        self.phi = np.sum(y == 1) / m
        self.mu0 = (np.dot(x.T, y == 0)) / (np.sum(y == 0))
        self.mu1 = (np.dot(x.T, y == 1)) / (np.sum(y == 1))
        self.sigma = np.dot((x - self.mu1).T * (y == 1), (x - self.mu1)) + np.dot((x - self.mu0).T * (y == 1), (x - self.mu0))
        self.sigma = self.sigma / m
        
        self.theta = np.zeros((n+1, 1))
        theta = np.dot(np.linalg.inv(self.sigma), self.mu1 - self.mu0)
        theta0 = 0.5 * np.dot(self.mu0.T, np.dot(np.linalg.inv(self.sigma), self.mu0)) 
        theta0 -= 0.5 * np.dot(self.mu1.T, np.dot(np.linalg.inv(self.sigma), self.mu1)) 
        theta0 += np.log((1-self.phi)/self.phi)
        self.theta[0] = theta0
        self.theta[1:] = np.reshape(theta, (-1,1))
        self.theta0 = theta0
        self.theta1 = theta
        print(f'Fitting result: theta is {self.theta}')
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(np.dot(x, self.theta1) + self.theta0)))
        # *** END CODE HERE
