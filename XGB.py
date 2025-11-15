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

    targetcol = 'label'
    featurecols = [c for c in dftrain.columns if c != targetcol]

    Xtrain = dftrain[featurecols].to_numpy(dtype=float)
    ytrain = dftrain[targetcol].to_numpy(dtype=int)

    Xval = dfval[featurecols].to_numpy(dtype=float)
    yval = dfval[targetcol].to_numpy(dtype=int)

    return Xtrain, ytrain, Xval, yval


# ---------- Core helpers ----------
def _sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _soft_threshold(g, alpha):
    return np.sign(g) * np.maximum(np.abs(g) - alpha, 0.0)


class _TreeNode:
    __slots__ = ("is_leaf", "weight", "feature", "threshold", "default_left", "left", "right")

    def __init__(self, is_leaf=True, weight=0.0, feature=None, threshold=None,
                 default_left=True, left=None, right=None):
        self.is_leaf = is_leaf
        self.weight = weight
        self.feature = feature
        self.threshold = threshold
        self.default_left = default_left
        self.left = left
        self.right = right

    def predict(self, X):
        if self.is_leaf:
            return np.full(X.shape[0], self.weight, dtype=float)
        xj = X[:, self.feature]
        isnan = np.isnan(xj)
        le = xj <= self.threshold
        if self.default_left:
            L = isnan | (le & ~isnan)
            R = (~isnan) & (~le)
        else:
            L = (~isnan) & le
            R = isnan | ((~isnan) & (~le))
        out = np.empty(X.shape[0], dtype=float)
        if L.any():
            out[L] = self.left.predict(X[L])
        if R.any():
            out[R] = self.right.predict(X[R])
        return out


# ---------- Binary XGBoost-style classifier (logistic loss) ----------
class XGBoostClassifier:
    """
    XGBoost-style **binary** classifier (logistic loss) with:
      - second-order (Newton) leaf updates (L1/L2)
      - histogram split finder using Hessian-weighted quantile bins
      - learned default direction for missing values
    """

    def __init__(self,
                 n_estimators=150,
                 learning_rate=0.3,
                 max_depth=6,
                 reg_lambda=1.0,
                 reg_alpha=0.0,
                 gamma=0.0,
                 min_child_weight=1.0,
                 subsample=1.0,
                 colsample_bytree=0.1,
                 n_bins=33,
                 missing_value=0,
                 base_score="auto",
                 random_state=42):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.n_bins = int(n_bins)
        self.missing_value = missing_value
        self.base_score = base_score
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._trees = []
        self._base_margin = 0.0

    # ---------- Public API ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float).copy()
        y = np.asarray(y, dtype=float)
        n_rows, n_cols = X.shape

        # Treat specific value as missing
        if self._is_specified_number(self.missing_value):
            X[X == self.missing_value] = np.nan

        # Base score (prior)
        if self.base_score == "auto":
            p0 = np.clip(y.mean(), 1e-15, 1 - 1e-15)
        else:
            p0 = np.clip(float(self.base_score), 1e-15, 1 - 1e-15)
        self._base_margin = np.log(p0 / (1 - p0))

        logits = np.full(n_rows, self._base_margin, dtype=float)
        self._trees.clear()

        all_feats = np.arange(n_cols)

        for _ in range(self.n_estimators):
            # Row subsampling
            if self.subsample < 1.0:
                m = max(1, int(self.subsample * n_rows))
                row_idx = self._rng.choice(n_rows, size=m, replace=False)
            else:
                row_idx = np.arange(n_rows)

            # Column subsampling
            if self.colsample_bytree < 1.0:
                f = max(1, int(self.colsample_bytree * n_cols))
                feat_idx = np.sort(self._rng.choice(all_feats, size=f, replace=False))
            else:
                feat_idx = all_feats

            # Gradients/Hessians (only on sampled rows)
            p = _sigmoid(logits[row_idx])
            g = p - y[row_idx]
            h = np.clip(p * (1 - p), 1e-12, None)

            # Build Hessian-weighted quantile edges for selected features
            edges = self._compute_weighted_bins(X[row_idx][:, feat_idx], h, self.n_bins)

            # Pre-bin features
            B, nbins, miss_bin = self._prebin_bmatrix(X[row_idx][:, feat_idx], edges)

            # Build tree using binned matrix
            lo, hi = 0, B.shape[0]
            root = self._build_tree_binned(B, g, h, lo, hi, 0, feat_idx, edges, nbins, miss_bin)

            step = root.predict(X)  # uses float thresholds on full X
            logits += self.learning_rate * step
            self._trees.append(root)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self._is_specified_number(self.missing_value):
            X[X == self.missing_value] = np.nan
        logits = np.full(X.shape[0], self._base_margin, dtype=float)
        for t in self._trees:
            logits += self.learning_rate * t.predict(X)
        p1 = _sigmoid(logits)
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ---------- Private helpers ----------
    def _is_specified_number(self, v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    def _leaf_weight(self, G, H):
        return - _soft_threshold(G, self.reg_alpha) / (H + self.reg_lambda)

    def _leaf_score(self, G, H):
        S = _soft_threshold(G, self.reg_alpha)
        return 0.5 * (S * S) / (H + self.reg_lambda)

    def _compute_weighted_bins(self, Xs, h, n_bins):
        """
        Compute per-feature thresholds using Hessian-weighted quantiles.
        Returns: dict local_feature_index -> np.ndarray of thresholds (ascending).
        """
        m, f = Xs.shape
        bins = {}
        for j in range(f):
            xj = Xs[:, j]
            mask = ~np.isnan(xj)
            if mask.sum() < 2:
                bins[j] = np.array([], dtype=float)
                continue

            xv = xj[mask]
            w = h[mask]
            order = np.argsort(xv, kind="mergesort")
            xv = xv[order]
            w = w[order]

            unique_vals = np.unique(xv)
            if unique_vals.size < 2:
                bins[j] = np.array([], dtype=float)
                continue

            cw = np.cumsum(w)
            total = cw[-1]
            if total <= 0:
                bins[j] = np.array([], dtype=float)
                continue

            k = max(1, min(n_bins - 1, unique_vals.size - 1))
            qs = (np.arange(1, k + 1) / (k + 1)) * total
            idxs = np.searchsorted(cw, qs, side="left")
            thresh = np.unique(xv[idxs])
            bins[j] = thresh
        return bins

    def _prebin_bmatrix(self, Xs, edges):
        """
        Xs: (m, f) float with possible NaNs
        Returns:
          B: (m, f) uint8 bins
          nbins: dict local_j -> nb
          miss_bin: dict local_j -> missing bin index
        """
        m, f = Xs.shape
        B = np.empty((m, f), dtype=np.uint8)
        nbins = {}
        miss_bin = {}
        for j in range(f):
            ej = edges.get(j, None)
            if ej is None or ej.size == 0:
                nb = 2
                mb = 1
                xj = Xs[:, j]
                isna = np.isnan(xj)
                B[:, j] = 0
                B[isna, j] = mb
            else:
                nb = ej.size + 2     # +1 for upper non-missing bin, +1 for missing bin
                mb = nb - 1
                xj = Xs[:, j]
                isna = np.isnan(xj)
                bins = np.searchsorted(ej, xj, side="right").astype(np.uint16)
                B[:, j] = bins
                B[isna, j] = mb
            nbins[j] = nb
            miss_bin[j] = mb
        return B, nbins, miss_bin

    def _build_tree_binned(self, B, g, h, lo, hi, depth, feature_map, edges, nbins, miss_bin):
        G, H = g[lo:hi].sum(), h[lo:hi].sum()
        if depth >= self.max_depth:
            return _TreeNode(is_leaf=True, weight=self._leaf_weight(G, H))

        best = self._find_best_split_binned(B, g, h, lo, hi, feature_map, edges, nbins, miss_bin)
        if best is None:
            return _TreeNode(is_leaf=True, weight=self._leaf_weight(G, H))

        j_local, split_bin, default_left, GL, HL, GR, HR, parent_score, gain = best
        if gain <= 0.0:
            return _TreeNode(is_leaf=True, weight=self._leaf_weight(G, H))

        # Partition rows [lo:hi) in-place
        k = self._partition_in_place(B, g, h, lo, hi, j_local, split_bin, default_left, miss_bin[j_local])

        # Recurse
        left = self._build_tree_binned(B, g, h, lo, k, depth + 1, feature_map, edges, nbins, miss_bin)
        right = self._build_tree_binned(B, g, h, k, hi, depth + 1, feature_map, edges, nbins, miss_bin)

        # Threshold: be defensive if edges are empty
        ej = edges.get(j_local, None)
        if ej is None or ej.size == 0:
            # fallback: no valid threshold, make this a leaf instead
            return _TreeNode(is_leaf=True, weight=self._leaf_weight(G, H))

        sb = min(split_bin, len(ej) - 1)
        thr = float(ej[sb])

        return _TreeNode(
            is_leaf=False, weight=0.0,
            feature=feature_map[j_local], threshold=thr, default_left=default_left,
            left=left, right=right
        )

    def _find_best_split_binned(self, B, g, h, lo, hi, feature_map, edges, nbins, miss_bin):
        parent_score = self._leaf_score(g[lo:hi].sum(), h[lo:hi].sum())
        best_gain = -np.inf
        best = None

        _, f = B.shape
        for j in range(f):
            # skip features with no thresholds
            ej = edges.get(j, None)
            if ej is None or ej.size == 0:
                continue

            nb = nbins[j]
            mb = miss_bin[j]      # missing bin index (= nb-1)
            nb_real = nb - 1      # exclude missing bin for prefix sums

            bj = B[lo:hi, j]      # contiguous slice
            gv = g[lo:hi]
            hv = h[lo:hi]

            G_per = np.bincount(bj, weights=gv, minlength=nb).astype(float)
            H_per = np.bincount(bj, weights=hv, minlength=nb).astype(float)

            Gmiss, Hmiss = G_per[mb], H_per[mb]
            Greal = G_per[:nb_real]
            Hreal = H_per[:nb_real]
            if Greal.size == 0:
                continue

            Gp = np.cumsum(Greal)
            Hp = np.cumsum(Hreal)
            Gtot, Htot = Gp[-1], Hp[-1]

            for s in range(nb_real):
                GL = Gp[s]; HL = Hp[s]
                GR = Gtot - GL; HR = Htot - HL

                # Option A: send missing left
                GL_A, HL_A = GL + Gmiss, HL + Hmiss
                GR_A, HR_A = GR, HR
                gain_A = -np.inf
                if HL_A >= self.min_child_weight and HR_A >= self.min_child_weight:
                    gain_A = (self._leaf_score(GL_A, HL_A) +
                              self._leaf_score(GR_A, HR_A) - parent_score) - self.gamma

                # Option B: send missing right
                GL_B, HL_B = GL, HL
                GR_B, HR_B = GR + Gmiss, HR + Hmiss
                gain_B = -np.inf
                if HL_B >= self.min_child_weight and HR_B >= self.min_child_weight:
                    gain_B = (self._leaf_score(GL_B, HL_B) +
                              self._leaf_score(GR_B, HR_B) - parent_score) - self.gamma

                if gain_A > gain_B:
                    gain, default_left = gain_A, True
                    GLb, HLb, GRb, HRb = GL_A, HL_A, GR_A, HR_A
                else:
                    gain, default_left = gain_B, False
                    GLb, HLb, GRb, HRb = GL_B, HL_B, GR_B, HR_B

                if gain > best_gain:
                    best_gain = gain
                    split_bin = s
                    best = (j, split_bin, default_left, GLb, HLb, GRb, HRb, parent_score, gain)

        return best

    def _partition_in_place(self, B, g, h, lo, hi, j, split_bin, default_left, missing_bin):
        """
        In-place partition rows [lo:hi) so that 'left' is [lo:k), 'right' is [k:hi).
        """
        i, k = lo, hi - 1
        while i <= k:
            b = B[i, j]
            if b == missing_bin:
                go_left = default_left
            else:
                go_left = (b <= split_bin)
            if go_left:
                i += 1
            else:
                B[i, :], B[k, :] = B[k, :].copy(), B[i, :].copy()
                g[i], g[k] = g[k], g[i]
                h[i], h[k] = h[k], h[i]
                k -= 1
        return i


# ---------- One-vs-Rest multiclass wrapper ----------
class OneVsRestXGBClassifier:
    """
    One-vs-rest multi-class wrapper around XGBoostClassifier.
    Trains one binary model per class and predicts via argmax of
    positive-class probabilities.
    """

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.classes_ = None
        self.models_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = []
        for c in self.classes_:
            y_bin = (y == c).astype(float)
            clf = XGBoostClassifier(**self.xgb_params)
            clf.fit(X, y_bin)
            self.models_.append(clf)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        prob_pos = np.column_stack(
            [clf.predict_proba(X)[:, 1] for clf in self.models_]
        )  # (n_samples, n_classes)
        row_sums = prob_pos.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return prob_pos / row_sums

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        prob_pos = np.column_stack(
            [clf.predict_proba(X)[:, 1] for clf in self.models_]
        )
        idx = np.argmax(prob_pos, axis=1)
        return self.classes_[idx]


# ---------- Standalone training/testing ----------
def main(cm_path="cm_xgb_digits.png"):
    Xtrain, ytrain, Xval, yval = read_data()

    clf = OneVsRestXGBClassifier(
        n_estimators=80,      # tune this for time vs accuracy
        learning_rate=0.3,
        max_depth=6,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=0.1,
        n_bins=33,
        missing_value=0,
        base_score="auto",
        random_state=42,
    )

    t0 = perf_counter()
    clf.fit(Xtrain, ytrain)
    train_time = perf_counter() - t0

    preds = clf.predict(Xval)

    acc  = accuracy_score(yval, preds)
    prec = precision_score(yval, preds, average='macro', zero_division=0)
    rec  = recall_score(yval, preds, average='macro', zero_division=0)
    f1   = f1_score(yval, preds, average='macro', zero_division=0)

    print(f"Accuracy (OvR XGB):          {acc:.6f}")
    print(f"Macro Precision:             {prec:.6f}  Macro Recall: {rec:.6f}  Macro F1: {f1:.6f}")
    print(f"Train time (all OvR models): {train_time:.3f}s")

    classes = clf.classes_
    cm = confusion_matrix(yval, preds, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, values_format='d', colorbar=False
    )
    ax.set_title("XGBoost OvR — Confusion Matrix (digits 0–9)")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
