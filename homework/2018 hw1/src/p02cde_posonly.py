import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, t_train)
    pred_c = clf.predict(x_test)
    np.savetxt(pred_path_c, pred_c > 0.5, fmt='%.1f')

    util.plot(x_test, t_test, clf.theta, "output/p02c.png")

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    new_clf = LogisticRegression()
    new_clf.fit(x_train, y_train)
    pred_d = new_clf.predict(x_test)
    np.savetxt(pred_path_d, pred_d > 0.5, fmt='%.1f')

    util.plot(x_test, t_test, new_clf.theta, "output/p02d.png")

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    pred_e = new_clf.predict(x_valid)

    V_pos = np.sum(y_valid == 1)
    alpha = np.sum((new_clf.predict(x_valid)) * np.reshape(y_valid == 1, (-1,1))) / V_pos
    rescaled_theta = new_clf.theta.copy()
    rescaled_theta[0] += np.log(2/alpha - 1)
    pred_e = new_clf.predict(x_test) / alpha
    np.savetxt(pred_path_e, pred_e > 0.5, fmt='%.1f')
    util.plot(x_test, t_test, rescaled_theta, "output/p02e.png")


    # *** END CODER HERE
