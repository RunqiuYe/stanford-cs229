import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%.1f')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        def h(theta, x):
            """
            calculate h based on theta and xi
            
            theta:  parameter. shape (n,)
            x:      all traning example inputs. shape (m, n)
            return: hypotheses for all traning example. shape (m,)
            """
            return 1 / (1 + np.exp(- np.dot(x, theta)))
        
        def gradient(theta, x, y):
            """
            calculate the gradient vector of J wrt to theta
            
            theta:  shape(n,)
            x:      all training example inputs. shape (m, n)
            y:      all training example labels. shape (m,)
            return: gradient of J. shape (n,)
            """
            m, n = x.shape
            y = np.reshape(y, (m, 1))
            return - 1 / m * np.dot(x.T, y - h(theta, x))
        
        def hessian(theta, x):
            """
            calculate the hessian matrix of J wrt to theta
            
            theta:  shape(n,)
            x:      all training example inputs. shape (m, n)
            return: hessian matrix of J. shape (n,n)
            """
            m, n = x.shape
            h_theta_x = np.reshape(h(theta, x), (-1, 1))
            return 1 / m * np.dot(x.T, h_theta_x * (1-h_theta_x) * x)
        
        def next_theta(theta, x, y):
            return theta - np.dot(np.linalg.inv(hessian(theta, x)), gradient(theta, x, y))
        
        m, n = x.shape
        self.theta = np.zeros((n, 1))
        
        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(abs(old_theta - new_theta)) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)
        self.theta = new_theta
        print(f'Fitting result: theta is {self.theta}')

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(- np.dot(x, self.theta)))
        # *** END CODE HERE ***