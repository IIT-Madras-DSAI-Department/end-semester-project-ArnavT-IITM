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

# ---------------- Data loader ----------------

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    targetcol = 'label'
    featurecols = [c for c in dftrain.columns if c != targetcol]

    Xtrain = dftrain[featurecols].to_numpy(dtype=float)
    ytrain = dftrain[targetcol].to_numpy(dtype=int)

    Xval = dfval[featurecols].to_numpy(dtype=float)
    yval = dfval[targetcol].to_numpy(dtype=int)

    return (Xtrain, ytrain, Xval, yval)


# ---------------- Optimized KNN with PCA ----------------

class KNNClassifier:
    """
    K-Nearest Neighbors with:
      - pixel normalization
      - PCA dimensionality reduction
      - squared Euclidean distance
      - distance-weighted voting
      - chunked prediction for memory efficiency

    Designed for MNIST-like data (digits 0–9).
    """

    def __init__(self,
                 n_neighbors=5,
                 n_components=60,
                 normalize_pixels=True,
                 use_pca=True,
                 chunk_size=256):
        """
        n_neighbors   : k in KNN
        n_components  : PCA components to keep (ignored if use_pca=False)
        normalize_pixels : divide by 255 if values look like 0..255
        use_pca       : whether to apply PCA before KNN
        chunk_size    : number of test samples per batch at prediction
        """
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.normalize_pixels = bool(normalize_pixels)
        self.use_pca = bool(use_pca)
        self.chunk_size = int(chunk_size)

        # Fitted attributes
        self.X_train_ = None      # (n_train, d_reduced)
        self.y_train_ = None      # (n_train,)
        self.classes_ = None      # (n_classes,)

        self.mean_ = None         # (n_features,)
        self.components_ = None   # (n_components, n_features) if PCA used

    # ----- Internal preprocessing -----

    def _normalize_pixels(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.normalize_pixels and X.max() > 1.0:
            X = X / 255.0
        return X

    def _fit_preprocess(self, X):
        """
        Normalize pixels, compute global mean, optionally PCA.
        Returns transformed training data.
        """
        X = self._normalize_pixels(X)

        # Center
        self.mean_ = X.mean(axis=0, dtype=np.float32)
        X_centered = X - self.mean_

        if not self.use_pca:
            # no PCA, work directly in pixel space
            self.components_ = None
            return X_centered.astype(np.float32)

        # PCA via SVD on centered data: X_centered = U S Vt
        # Keep top n_components from Vt (principal directions).
        # This is on 784 dims, so it's quite manageable.
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Vt: (n_features, n_features), take first n_components
        self.components_ = Vt[:self.n_components].astype(np.float32)  # (n_components, d)

        # Project into PCA space: (n_samples, n_components)
        X_reduced = X_centered @ self.components_.T
        return X_reduced.astype(np.float32)

    def _predict_preprocess(self, X):
        """
        Apply same normalization + centering + PCA as in fit.
        """
        X = self._normalize_pixels(X)
        X_centered = X - self.mean_

        if self.use_pca and self.components_ is not None:
            X_reduced = X_centered @ self.components_.T
            return X_reduced.astype(np.float32)
        else:
            return X_centered.astype(np.float32)

    # ----- Fitting -----

    def fit(self, X, y):
        """
        Training is: preprocess + PCA + store.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        Xp = self._fit_preprocess(X)
        self.X_train_ = Xp
        self.y_train_ = y
        self.classes_ = np.unique(y)

        if self.n_neighbors > self.X_train_.shape[0]:
            raise ValueError("n_neighbors cannot be larger than number of training samples")

        return self

    # ----- Distance computation -----

    def _block_squared_distances(self, X_block):
        """
        Compute squared Euclidean distances between:
          X_block: (b, d) and X_train_: (n_train, d)

        Uses: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        """
        Xb = X_block.astype(np.float32)
        Xt = self.X_train_

        xb2 = np.sum(Xb * Xb, axis=1, keepdims=True)  # (b, 1)
        xt2 = np.sum(Xt * Xt, axis=1)                 # (n_train,)

        cross = Xb @ Xt.T                              # (b, n_train)
        dists = xb2 + xt2[None, :] - 2.0 * cross
        np.maximum(dists, 0.0, out=dists)             # numeric safety

        return dists

    # ----- Predict proba & predict -----

    def predict_proba(self, X):
        """
        Returns probability distribution over classes per sample, based on
        distance-weighted neighbor voting.

        p(class c) = sum_{neighbors i with label c} w_i / sum_i w_i
        where w_i = 1 / (dist_i + eps)
        """
        if self.X_train_ is None:
            raise RuntimeError("KNNClassifier is not fitted yet.")

        Xp = self._predict_preprocess(X)
        n_test = Xp.shape[0]
        n_classes = len(self.classes_)

        # map labels -> indices in [0, n_classes)
        label_to_idx = {c: idx for idx, c in enumerate(self.classes_)}

        proba = np.zeros((n_test, n_classes), dtype=np.float32)
        k = self.n_neighbors
        cs = self.chunk_size
        eps = 1e-8

        for start in range(0, n_test, cs):
            end = min(start + cs, n_test)
            X_block = Xp[start:end]

            # (b, n_train) distances
            dists = self._block_squared_distances(X_block)

            # indices of k nearest neighbors per row
            nn_idx = np.argpartition(dists, k, axis=1)[:, :k]  # (b, k)
            nn_dists = np.take_along_axis(dists, nn_idx, axis=1)  # (b, k)
            nn_labels = self.y_train_[nn_idx]                    # (b, k)

            # convert squared distances to weights: w = 1 / (sqrt(d2) + eps)
            # using sqrt for more intuitive scaling (could also use 1/(d2+eps))
            nn_dists = np.sqrt(nn_dists, dtype=np.float32)
            weights = 1.0 / (nn_dists + eps)    # (b, k)

            b = nn_labels.shape[0]
            block_proba = np.zeros((b, n_classes), dtype=np.float32)

            # accumulate weights per class
            for i in range(b):
                labs = nn_labels[i]   # (k,)
                ws   = weights[i]     # (k,)
                for lbl, w in zip(labs, ws):
                    j = label_to_idx[lbl]
                    block_proba[i, j] += w

            # normalize row-wise
            row_sums = block_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            block_proba /= row_sums

            proba[start:end] = block_proba

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ---------------- Standalone test on MNIST ----------------

def main(cm_path="cm_knn_pca_digits.png"):
    Xtrain, ytrain, Xval, yval = read_data()

    clf = KNNClassifier(
        n_neighbors=5,
        n_components=60,
        normalize_pixels=True,
        use_pca=True,
        chunk_size=256,
    )

    t0 = perf_counter()
    clf.fit(Xtrain, ytrain)
    train_time = perf_counter() - t0
    print(f"KNN (with PCA) training time: {train_time:.3f}s")

    t1 = perf_counter()
    preds = clf.predict(Xval)
    pred_time = perf_counter() - t1
    print(f"KNN (with PCA) prediction time on validation: {pred_time:.3f}s")

    acc = accuracy_score(yval, preds)
    p   = precision_score(yval, preds, average='macro', zero_division=0)
    r   = recall_score(yval, preds, average='macro', zero_division=0)
    f1  = f1_score(yval, preds, average='macro', zero_division=0)

    print(f"KNN+PCA Accuracy:        {acc:.6f}")
    print(f"KNN+PCA Macro Precision: {p:.6f}  Macro Recall: {r:.6f}  Macro F1: {f1:.6f}")

    # Confusion matrix
    labels = np.unique(np.concatenate([ytrain, yval]))
    cm = confusion_matrix(yval, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, values_format='d', colorbar=False
    )
    ax.set_title("KNN + PCA — Confusion Matrix (digits 0–9)")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved KNN+PCA confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
