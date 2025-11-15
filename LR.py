import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------- Data loader ----------
def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)
    
    # Predict the actual digit (0–9)
    targetcol = 'label'
    featurecols = [c for c in dftrain.columns if c != targetcol]

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]
    Xval   = dfval[featurecols]
    yval   = dfval[targetcol]
    return (Xtrain, ytrain, Xval, yval)


# ---------- Core helper: binary logistic regression ----------
def sigmoid(z, clip=35.0):
    """
    Numerically stable sigmoid with clipping.
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -clip, clip)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_regression_binary(x, y, learning_rate=0.1, n_epochs=15, batch_size=256, rng=None):
    """
    Mini-batch gradient descent for **binary** logistic regression.
    y in {0,1}.
    Returns theta of shape (n_features+1, 1) including bias.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    # add bias column
    x_b = np.c_[np.ones((x.shape[0], 1)), x]  # (m, n+1)
    m, n_plus_1 = x_b.shape
    theta = np.zeros((n_plus_1, 1), dtype=float)

    if rng is None:
        rng = np.random.default_rng(42)

    for _ in range(n_epochs):
        idx = rng.permutation(m)
        x_b_shuf = x_b[idx]
        y_shuf   = y[idx]

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            xb = x_b_shuf[start:end]
            yb = y_shuf[start:end]

            p = sigmoid(xb @ theta)                 # (batch, 1)
            grad = (xb.T @ (p - yb)) / (end - start)
            theta -= learning_rate * grad

    return theta  # (n_features+1, 1)


# ---------- OvR Logistic Regression CLASS ----------
class LogisticRegressionOVR:
    """
    One-vs-rest multinomial logistic regression using mini-batch GD.
    - Trains one binary LR per class c: y_c = 1[y==c]
    - Uses sigmoid on each class score and argmax over classes at prediction.
    """

    def __init__(self,
                 learning_rate=0.1,
                 n_epochs=15,
                 batch_size=256,
                 standardize=True,
                 random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.standardize = standardize
        self.random_state = random_state

        self.classes_ = None       # shape (K,)
        self.theta_ = None         # shape (n_features+1, K)
        self.mu_ = None            # feature means (for scaling)
        self.sigma_ = None         # feature stds (for scaling)
        self._rng = np.random.default_rng(random_state)

    # ----- internal preprocessing -----
    def _fit_preprocess(self, X):
        """
        Convert X to numpy, scale pixels, and optional standardization.
        Stores mu_ and sigma_ if standardize=True.
        """
        X = np.asarray(X, dtype=float)

        # Pixels: scale to [0,1] if they look like 0..255
        if X.max() > 1.0:
            X = X / 255.0

        if self.standardize:
            self.mu_ = X.mean(axis=0)
            self.sigma_ = X.std(axis=0) + 1e-8
            X = (X - self.mu_) / self.sigma_
        else:
            self.mu_ = None
            self.sigma_ = None

        return X

    def _predict_preprocess(self, X):
        """
        Apply the same scaling/std as in fit().
        """
        X = np.asarray(X, dtype=float)
        if X.max() > 1.0:
            X = X / 255.0

        if self.standardize and self.mu_ is not None and self.sigma_ is not None:
            X = (X - self.mu_) / self.sigma_

        return X

    # ----- public API -----
    def fit(self, X, y):
        """
        Train OvR logistic regression.
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,) with integer class labels (e.g. digits 0–9)
        """
        X = self._fit_preprocess(X)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        K = self.classes_.shape[0]
        n_features = X.shape[1]

        # theta_: (n_features+1, K)
        self.theta_ = np.zeros((n_features + 1, K), dtype=float)

        for k, c in enumerate(self.classes_):
            y_bin = (y == c).astype(int)
            theta_c = _logistic_regression_binary(
                X,
                y_bin,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                rng=self._rng,
            )  # (n_features+1, 1)
            self.theta_[:, k:k+1] = theta_c

        return self

    def predict_proba(self, X):
        """
        Return OvR probabilities (not softmax, but per-class sigmoids),
        normalized to sum to 1 per sample.
        shape: (n_samples, n_classes)
        """
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")

        Xp = np.asarray(self._predict_preprocess(X), dtype=float)
        X_b = np.c_[np.ones((Xp.shape[0], 1)), Xp]  # (n_samples, n_features+1)

        # scores: (n_samples, K)
        scores = X_b @ self.theta_  # linear logits for each class
        probs = sigmoid(scores)     # apply sigmoid to each class logit

        # normalize row-wise
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        probs = probs / row_sums

        return probs

    def predict(self, X):
        """
        Predict class labels via argmax over OvR class scores.
        """
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")

        Xp = np.asarray(self._predict_preprocess(X), dtype=float)
        X_b = np.c_[np.ones((Xp.shape[0], 1)), Xp]

        scores = X_b @ self.theta_  # (n_samples, K)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


# ---------- Standalone training/testing ----------
def main(cm_path="cm_lr_digits.png"):
    X, y, Xval, yval = read_data()

    # to numpy
    Xtr = X.values
    ytr = y.values
    Xva = Xval.values
    yva = yval.values

    clf = LogisticRegressionOVR(
        learning_rate=0.1,
        n_epochs=15,
        batch_size=256,
        standardize=True,
        random_state=42,
    )

    t0 = perf_counter()
    clf.fit(Xtr, ytr)
    train_time = perf_counter() - t0

    preds = clf.predict(Xva)

    acc  = accuracy_score(yva, preds)
    prec = precision_score(yva, preds, average='macro', zero_division=0)
    rec  = recall_score(yva, preds, average='macro', zero_division=0)
    f1   = f1_score(yva, preds, average='macro', zero_division=0)

    print(f"Accuracy:        {acc:.6f}")
    print(f"Macro Precision: {prec:.6f}  Macro Recall: {rec:.6f}  Macro F1: {f1:.6f}")
    print(f"Train time:      {train_time:.3f}s")

    # Confusion matrix
    classes = clf.classes_
    cm = confusion_matrix(yva, preds, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, values_format='d', colorbar=False
    )
    ax.set_title("Logistic Regression (OvR) — Confusion Matrix (digits 0–9)")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
