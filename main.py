import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import KFold

from KNN import KNNClassifier
from LR import LogisticRegressionOVR
from XGB import OneVsRestXGBClassifier


# ---------- Data loader ----------
def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    dftrain.drop('even', axis = 1, inplace=True)
    dfval.drop('even', axis = 1, inplace=True)
    
    targetcol = 'label'
    featurecols = [c for c in dftrain.columns if c != targetcol]

    Xtrain = dftrain[featurecols].to_numpy(dtype=float)
    ytrain = dftrain[targetcol].to_numpy(dtype=int)

    Xval = dfval[featurecols].to_numpy(dtype=float)
    yval = dfval[targetcol].to_numpy(dtype=int)

    return Xtrain, ytrain, Xval, yval


# ---------- Utility ----------
def align_proba(proba, source_classes, target_classes):
    """
    Reorder columns of proba so they match target_classes.
    proba: (n_samples, len(source_classes))
    """
    source_classes = np.asarray(source_classes)
    target_classes = np.asarray(target_classes)
    idx_map = {c: i for i, c in enumerate(source_classes)}
    n_samples = proba.shape[0]
    out = np.zeros((n_samples, len(target_classes)), dtype=float)
    for j, c in enumerate(target_classes):
        if c in idx_map:
            out[:, j] = proba[:, idx_map[c]]
    return out


# ---------- Stacking ----------
def main():
    overall_start = perf_counter()

    Xtrain, ytrain, Xval, yval = read_data()
    classes = np.unique(ytrain)
    C = len(classes)
    n_train = Xtrain.shape[0]

    print(f"Train shape: {Xtrain.shape}, Val shape: {Xval.shape}, Classes: {classes}")

    # Base model constructors (so we can re-create per fold and for full train)
    def make_knn():
        return KNNClassifier(
            n_neighbors=3,
            n_components=60,
            normalize_pixels=True,
            use_pca=True,
            chunk_size=256,
        )

    def make_lr():
        return LogisticRegressionOVR(
            learning_rate=0.1,
            n_epochs=15,
            batch_size=256,
            standardize=True,
            random_state=42,
        )

    def make_xgb():
        # Slightly lighter XGB to hit <5 min:
        #  - n_estimators: 24 (was 30)
        #  - depth 5, subsample 0.8 as before
        return OneVsRestXGBClassifier(
            n_estimators=21,
            learning_rate=0.3,
            max_depth=5,
            reg_lambda=1.0,
            reg_alpha=0.0,
            gamma=0.0,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.05,
            n_bins=33,
            missing_value=0,
            base_score="auto",
            random_state=42,
        )

    # Out-of-fold probabilities for each base model
    oof_knn = np.zeros((n_train, C), dtype=float)
    oof_lr  = np.zeros((n_train, C), dtype=float)
    oof_xgb = np.zeros((n_train, C), dtype=float)

    # 2-fold stacking for speed
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    print("Generating out-of-fold probabilities for stacking...")

    fold_idx = 1
    t_stack_start = perf_counter()
    for tr_idx, val_idx in kf.split(Xtrain):
        print(f"\n--- Fold {fold_idx} ---")
        fold_idx += 1

        X_tr, X_fval = Xtrain[tr_idx], Xtrain[val_idx]
        y_tr, y_fval = ytrain[tr_idx], ytrain[val_idx]

        # ---- KNN ----
        knn = make_knn()
        t0 = perf_counter()
        knn.fit(X_tr, y_tr)
        p_knn = knn.predict_proba(X_fval)
        p_knn = align_proba(p_knn, knn.classes_, classes)
        oof_knn[val_idx] = p_knn
        t_knn = perf_counter() - t0
        print(f"KNN fold fit+proba time: {t_knn:.3f}s")

        # ---- LR OvR ----
        lr = make_lr()
        t0 = perf_counter()
        lr.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_fval)
        p_lr = align_proba(p_lr, lr.classes_, classes)
        oof_lr[val_idx] = p_lr
        t_lr = perf_counter() - t0
        print(f"LR fold fit+proba time:  {t_lr:.3f}s")

        # ---- XGB OvR ----
        xgb = make_xgb()
        t0 = perf_counter()
        xgb.fit(X_tr, y_tr)
        p_xgb = xgb.predict_proba(X_fval)
        p_xgb = align_proba(p_xgb, xgb.classes_, classes)
        oof_xgb[val_idx] = p_xgb
        t_xgb = perf_counter() - t0
        print(f"XGB fold fit+proba time: {t_xgb:.3f}s")

    t_stack_oof = perf_counter() - t_stack_start
    print(f"\nTotal OOF generation time: {t_stack_oof:.3f}s")

    # Meta-features: concatenate all 3 models' OOF probs -> shape (n_train, 3*C)
    Z_train = np.concatenate([oof_knn, oof_lr, oof_xgb], axis=1)

    # Meta-learner: LR on top (on 30-dim prob features)
    meta = LogisticRegressionOVR(
        learning_rate=0.1,
        n_epochs=10,       # features are already strong
        batch_size=256,
        standardize=True,
        random_state=1337,
    )

    print("\nTraining meta-learner on stacked features...")
    t0 = perf_counter()
    meta.fit(Z_train, ytrain)
    t_meta = perf_counter() - t0
    print(f"Meta-learner fit time: {t_meta:.3f}s")

    # ---- Retrain base models on full training set ----
    print("\nTraining base models on full training data...")

    knn_full = make_knn()
    t0 = perf_counter()
    knn_full.fit(Xtrain, ytrain)
    t_knn_full = perf_counter() - t0
    print(f"KNN full fit time: {t_knn_full:.3f}s")

    lr_full = make_lr()
    t0 = perf_counter()
    lr_full.fit(Xtrain, ytrain)
    t_lr_full = perf_counter() - t0
    print(f"LR full fit time:  {t_lr_full:.3f}s")

    xgb_full = make_xgb()
    t0 = perf_counter()
    xgb_full.fit(Xtrain, ytrain)
    t_xgb_full = perf_counter() - t0
    print(f"XGB full fit time: {t_xgb_full:.3f}s")

    # ---- Build meta features for validation set ----
    print("\nComputing base model probabilities on validation set...")

    p_knn_val = knn_full.predict_proba(Xval)
    p_knn_val = align_proba(p_knn_val, knn_full.classes_, classes)

    p_lr_val = lr_full.predict_proba(Xval)
    p_lr_val = align_proba(p_lr_val, lr_full.classes_, classes)

    p_xgb_val = xgb_full.predict_proba(Xval)
    p_xgb_val = align_proba(p_xgb_val, xgb_full.classes_, classes)

    Z_val = np.concatenate([p_knn_val, p_lr_val, p_xgb_val], axis=1)

    # ---- Meta prediction ----
    print("\nEvaluating stacked ensemble on validation set...")
    preds = meta.predict(Z_val)

    acc  = accuracy_score(yval, preds)
    prec = precision_score(yval, preds, average='macro', zero_division=0)
    rec  = recall_score(yval, preds, average='macro', zero_division=0)
    f1   = f1_score(yval, preds, average='macro', zero_division=0)

    print("\nStacked Ensemble (KNN + LR + XGB) — FAST CONFIG v2")
    print(f"Accuracy:        {acc:.6f}")
    print(f"Macro Precision: {prec:.6f}  Macro Recall: {rec:.6f}  Macro F1: {f1:.6f}")

    # Confusion matrix
    cm = confusion_matrix(yval, preds, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, values_format='d', colorbar=False
    )
    ax.set_title("Stacked Ensemble (fast v2) — Confusion Matrix (digits 0–9)")
    fig.tight_layout()
    fig.savefig("cm_stacked_knn_lr_xgb_fast_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion matrix to cm_stacked_knn_lr_xgb_fast_v2.png")

    overall_time = perf_counter() - overall_start
    print(f"\nTotal end-to-end time (stacking + full models + eval): {overall_time:.3f}s")


if __name__ == "__main__":
    main()
