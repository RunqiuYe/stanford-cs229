import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    minMSE = 10**6
    bestTau = tau_values[0]
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau=tau)
        lwr.fit(x_train, y_train)
        y_pred = lwr.predict(x_valid)
        y_pred = np.reshape(y_pred, (-1,1))
        y_valid = np.reshape(y_valid, (-1,1))
        mse = np.mean((y_pred - y_valid)**2)
        print(f'tau, mse: {tau}, {mse}')
        if mse < minMSE:
            bestTau = tau
            minMSE = mse

        plt.figure()
        plt.plot(x_train, y_train, 'bx')
        plt.plot(x_valid, y_pred, 'ro')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/p05c_tau_{tau}.png')
    
    print(f'Best tau: {bestTau}')
    # Fit a LWR model with the best tau value
    lwr = LocallyWeightedLinearRegression(tau=bestTau)
    lwr.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = lwr.predict(x_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f'MSE for the test set: {mse}')
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    plt.figure()
    plt.plot(x_test, y_test, 'bx')
    plt.plot(x_test, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05c_test.png')
    # *** END CODE HERE ***
