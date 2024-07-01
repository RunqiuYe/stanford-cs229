import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    poi_model = PoissonRegression(step_size=lr, eps=1e-5)
    poi_model.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    y_pred = poi_model.predict(x_valid)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        alpha = self.step_size
        theta = np.zeros((n,1))
        next_theta = theta + alpha * np.dot(x.T, np.reshape(y, (-1,1)) - np.exp(np.dot(x, theta))) / m
        
        while (np.linalg.norm(theta - next_theta) > self.eps):
            theta = next_theta
            next_theta = theta + alpha * np.dot(x.T, np.reshape(y, (-1,1)) - np.exp(np.dot(x, theta))) / m
        self.theta = next_theta
        # print(f'Fitting results: theta is {self.theta}')


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***
