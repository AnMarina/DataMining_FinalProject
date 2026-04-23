"""
ssl_methods.py — Semi-supervised learning algorithms:

  1. Vanilla Pseudo-Labeling  (Lee, 2013)
  2. UPS  (Rizve et al., 2021)
  3. Adaptive UF-SSL  (proposed)
  4. Adaptive UF-SSL + Entropy-Weighted Loss  (Stage 2)
  5. Cost-Sensitive Asymmetric UF-SSL  (Stage 6 / Champion)
  6. FNR-Driven Defer Gate UF-SSL  (Stage 7)
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.model import (
    make_mlp,
    train_model,
    train_model_weighted,
    train_model_cost_sensitive,
    train_model_joint_weighted,
    predict_proba_nn,
    mc_dropout_probs,
    predictive_entropy,
    get_metrics_nn,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def get_entropy_threshold(current_round, max_rounds, h_min=0.01, h_max=0.50, mode='linear'):
    """
    Adaptive entropy admission threshold tau_H(t).

    This threshold controls how much uncertainty is allowed when accepting
    pseudo-labels. Lower values are stricter; higher values are more permissive.

    Parameters
    ----------
    mode : 'linear' | 'cosine' | 'step'
        Controls how the entropy threshold evolves over rounds.
    """
    progress = current_round / max(1, max_rounds - 1)

    if mode == 'linear':
        return h_min + (h_max - h_min) * progress
    elif mode == 'cosine':
        return h_min + 0.5 * (h_max - h_min) * (1 - np.cos(np.pi * progress))
    elif mode == 'step':
        return h_min if progress < 0.5 else h_max
    else:
        raise ValueError(f"Unsupported decay mode: {mode!r}")


# ---------------------------------------------------------------------------
# 1. Vanilla Pseudo-Labeling  (Lee, 2013)
# ---------------------------------------------------------------------------

def run_vanilla_pl(X_lab_sc, y_labeled, X_unlab_sc, X_test_sc, y_test,
                   seed=42, tau=0.95, n_epochs=50, max_rounds=10):
    """
    Basic pseudo-labeling:
    - train on labeled data
    - predict on unlabeled pool
    - accept samples whose max class probability >= tau
    - add them to training data and repeat
    """
    model = make_mlp(seed)

    # Current supervised training set (starts with labeled data only)
    X_train = X_lab_sc.copy()
    y_train = y_labeled.copy()

    # Remaining unlabeled pool to mine pseudo-labels from
    X_pool = X_unlab_sc.copy()

    history = []
    final_metrics = None

    print(f"\n  tau={tau} | epochs/round={n_epochs} | max_rounds={max_rounds}")
    print(f"  Starting labeled pool: {len(X_train)} samples\n")

    for rnd in range(max_rounds):
        # Train on everything currently trusted enough to be in the train set
        train_model(model, X_train, y_train, n_epochs=n_epochs)

        if len(X_pool) == 0:
            print(f"  Round {rnd+1}: unlabeled pool exhausted.")
            final_metrics = get_metrics_nn(model, X_test_sc, y_test)
            break

        # Predict class probabilities for the unlabeled pool
        probs = predict_proba_nn(model, X_pool)
        max_conf = probs.max(axis=1)
        pseudo_y = probs.argmax(axis=1)

        # Keep only high-confidence pseudo-labels
        mask = max_conf >= tau
        n_sel = int(mask.sum())

        if n_sel == 0:
            print(f"  Round {rnd+1}: no samples above tau={tau}. Stopping early.")
            final_metrics = get_metrics_nn(model, X_test_sc, y_test)
            break

        # Add accepted pseudo-labeled samples to the training set
        X_train = np.vstack([X_train, X_pool[mask]])
        y_train = np.concatenate([y_train, pseudo_y[mask]])

        # Remove accepted samples from the unlabeled pool
        X_pool = X_pool[~mask]

        # Small retrain pass so logged metrics reflect the updated train set
        train_model(model, X_train, y_train, n_epochs=max(10, n_epochs // 3))

        m = get_metrics_nn(model, X_test_sc, y_test)
        final_metrics = m

        history.append({
            'round': rnd + 1,
            'selected': n_sel,
            'train_size': len(X_train),
            'pool_left': len(X_pool),
            'test_acc': m['accuracy'],
            'test_f1': m['f1'],
            'test_auc': m['auc'],
            'test_fnr': m['fnr'],
        })

        print(f"  Round {rnd+1:2d}: selected={n_sel:3d}, train_size={len(X_train):3d}, "
              f"pool_left={len(X_pool):3d}, acc={m['accuracy']:.4f}, fnr={m['fnr']:.4f}")

    # If the loop finishes normally, still compute final test metrics
    if final_metrics is None:
        final_metrics = get_metrics_nn(model, X_test_sc, y_test)

    return model, history, final_metrics


# ---------------------------------------------------------------------------
# 2. UPS  (Rizve et al., 2021)
# ---------------------------------------------------------------------------

def run_ups(X_lab_sc, y_labeled, X_unlab_sc, X_test_sc, y_test,
            seed=42, tau=0.95, sigma=0.05, t_mc=30, n_epochs=50, max_rounds=10):
    """
    UPS adds a second safety filter on top of confidence:
    a sample must be both high-confidence and low-uncertainty under MC-Dropout.
    """
    model = make_mlp(seed)
    X_train = X_lab_sc.copy()
    y_train = y_labeled.copy()
    X_pool = X_unlab_sc.copy()
    history = []

    # Running totals used for summary stats across rounds
    total_conf_pass = 0
    total_dual_pass = 0
    total_uncert_rej = 0

    print(f"  tau={tau} | sigma={sigma} | T={t_mc} | max_rounds={max_rounds}")
    print(f"  Starting labeled pool: {len(X_train)} samples\n")

    for rnd in range(max_rounds):
        train_model(model, X_train, y_train, n_epochs=n_epochs)

        if len(X_pool) == 0:
            print(f"  Round {rnd+1}: unlabeled pool exhausted.")
            break

        # MC-Dropout gives multiple stochastic predictions for each unlabeled sample
        mc = mc_dropout_probs(model, X_pool, T=t_mc, seed=seed + rnd)
        mean_probs = mc.mean(axis=0)

        # Uncertainty proxy: std of positive-class probability across MC passes
        mc_std = mc[:, :, 1].std(axis=0)

        max_conf = mean_probs.max(axis=1)
        pseudo_y = mean_probs.argmax(axis=1)

        # First filter: standard confidence threshold
        conf_mask = max_conf >= tau

        # Second filter: prediction must also be stable across MC passes
        dual_mask = conf_mask & (mc_std <= sigma)
        uncert_rej = int(conf_mask.sum() - dual_mask.sum())

        total_conf_pass += int(conf_mask.sum())
        total_dual_pass += int(dual_mask.sum())
        total_uncert_rej += uncert_rej

        n_sel = int(dual_mask.sum())
        if n_sel == 0:
            print(f"  Round {rnd+1}: no samples passed dual filter. Stopping early.")
            break

        X_train = np.vstack([X_train, X_pool[dual_mask]])
        y_train = np.concatenate([y_train, pseudo_y[dual_mask]])
        X_pool = X_pool[~dual_mask]

        train_model(model, X_train, y_train, n_epochs=max(10, n_epochs // 3))
        m = get_metrics_nn(model, X_test_sc, y_test)

        history.append({
            'round': rnd + 1,
            'conf_pass': int(conf_mask.sum()),
            'dual_pass': n_sel,
            'uncert_rejected': uncert_rej,
            'train_size': len(X_train),
            'pool_left': len(X_pool),
            'test_acc': m['accuracy'],
            'test_fnr': m['fnr'],
        })

        print(f"  Round {rnd+1:2d}: conf_pass={int(conf_mask.sum()):3d}, dual_pass={n_sel:3d}, "
              f"rej_by_sigma={uncert_rej:3d}, pool_left={len(X_pool):3d}, "
              f"acc={m['accuracy']:.4f}, fnr={m['fnr']:.4f}")

    final_metrics = get_metrics_nn(model, X_test_sc, y_test)
    rej_rate = total_uncert_rej / max(total_conf_pass, 1)

    print(f"\n  UPS dual-filter stats (all rounds summed):")
    print(f"    Conf-passing    : {total_conf_pass}")
    print(f"    Passed dual     : {total_dual_pass}")
    print(f"    Rejected by σ   : {total_uncert_rej}")
    print(f"    Rej rate        : {rej_rate*100:.1f}%")

    return model, history, final_metrics


# ---------------------------------------------------------------------------
# 3. Adaptive UF-SSL  (proposed baseline)
# ---------------------------------------------------------------------------

def run_adaptive_ufssl(X_lab_sc, y_labeled, X_unlab_sc, X_test_sc, y_test,
                       seed=42, tau=0.95, h_min=0.01, h_max=0.50,
                       n_epochs=50, max_rounds=10, schedule='linear',
                       max_add_per_round=50):
    """
    Accept pseudo-labels only if they are:
    - confident enough (max prob >= tau)
    - low-entropy enough (entropy <= tau_h)

    tau_h changes by round, letting the method gradually loosen or tighten
    the uncertainty gate depending on the chosen schedule.
    """
    np.random.seed(seed)

    model = make_mlp(seed)
    X_train = X_lab_sc.copy()
    y_train = y_labeled.copy()
    X_pool = X_unlab_sc.copy()
    history = []

    print(f"  tau={tau} | H_min={h_min} | H_max={h_max} | max_rounds={max_rounds}")
    print(f"  Starting labeled pool: {len(X_train)} samples\n")

    for rnd in range(max_rounds):
        train_model(model, X_train, y_train, n_epochs=n_epochs)

        if len(X_pool) == 0:
            print(f"  Round {rnd+1}: unlabeled pool exhausted.")
            break

        probs = predict_proba_nn(model, X_pool)
        max_conf = probs.max(axis=1)
        pseudo_y = probs.argmax(axis=1)
        entropy = predictive_entropy(probs)

        # Round-dependent entropy gate
        tau_h = get_entropy_threshold(rnd, max_rounds, h_min, h_max, mode=schedule)

        # Candidates must pass both confidence and entropy filters
        candidate_idx = np.where((max_conf >= tau) & (entropy <= tau_h))[0]

        if len(candidate_idx) == 0:
            print(f"  Round {rnd+1}: no samples passed dual filter. Stopping early.")
            break

        # Among valid candidates, prefer the lowest-entropy ones first
        order = np.argsort(entropy[candidate_idx])
        selected_idx = candidate_idx[order[:max_add_per_round]]

        mask = np.zeros(len(X_pool), dtype=bool)
        mask[selected_idx] = True
        n_sel = int(mask.sum())

        X_train = np.vstack([X_train, X_pool[mask]])
        y_train = np.concatenate([y_train, pseudo_y[mask]])
        X_pool = X_pool[~mask]

        # retrain after adding new pseudo-labels
        train_model(model, X_train, y_train, n_epochs=max(10, n_epochs // 3))

        m = get_metrics_nn(model, X_test_sc, y_test)
        history.append({
            'round': rnd + 1,
            'selected': n_sel,
            'tau_h': tau_h,
            'avg_entropy_selected': float(entropy[mask].mean()),
            'test_fnr': m['fnr'],
            'test_ece': m['ece']
        })

        print(f"  Round {rnd+1:2d}: tau_H={tau_h:.3f}, selected={n_sel:3d}, "
              f"pool_left={len(X_pool):3d}, fnr={m['fnr']:.4f}, ece={m['ece']:.4f}")

    return model, history, get_metrics_nn(model, X_test_sc, y_test)


# ---------------------------------------------------------------------------
# 4. Adaptive UF-SSL + Entropy-Weighted Loss  (Stage 2)
# ---------------------------------------------------------------------------

def run_ufssl_weighted(X_lab_sc, y_labeled, X_unlab_sc, X_test_sc, y_test,
                       seed=42, tau=0.95, h_min=0.01, h_max=0.50,
                       n_epochs=50, max_rounds=10, schedule='cosine'):
    """
    Same acceptance idea as Adaptive UF-SSL, but accepted pseudo-labels are
    not trusted equally. Lower-entropy pseudo-labels get higher training weight.
    """
    model = make_mlp(seed)

    X_train = X_lab_sc.copy()
    y_train = y_labeled.copy()

    # Start labeled examples with full weight 1.0
    w_train = np.ones(len(y_labeled), dtype=np.float32)

    X_pool = X_unlab_sc.copy()
    history = []

    for rnd in range(max_rounds):
        # Weighted training lets more reliable pseudo-labels matter more
        train_model_weighted(
            model,
            X_train,
            y_train,
            weights_np=w_train,
            n_epochs=n_epochs,
            seed=seed + rnd
        )

        if len(X_pool) == 0:
            print(f"  Round {rnd+1}: unlabeled pool exhausted.")
            break

        probs = predict_proba_nn(model, X_pool)
        max_conf = probs.max(axis=1)
        pseudo_y = probs.argmax(axis=1)
        entropy = predictive_entropy(probs)

        tau_h = get_entropy_threshold(rnd, max_rounds, h_min, h_max, mode=schedule)
        mask = (max_conf >= tau) & (entropy <= tau_h)
        n_sel = int(mask.sum())

        if n_sel == 0:
            print(f"  Round {rnd+1}: no samples passed weighted filter. Stopping early.")
            break

        # Convert entropy into a trust weight in [0, 1], so lower entropy -> larger weight
        w_pseudo = 1.0 - np.clip(entropy[mask] / max(h_max, 1e-8), 0.0, 1.0)

        X_train = np.vstack([X_train, X_pool[mask]])
        y_train = np.concatenate([y_train, pseudo_y[mask]])
        w_train = np.concatenate([w_train, w_pseudo.astype(np.float32)])
        X_pool = X_pool[~mask]

        # retrain after adding new pseudo-labels. This makes the logged metrics reflect the updated train set
        train_model_weighted(
            model,
            X_train,
            y_train,
            weights_np=w_train,
            n_epochs=max(10, n_epochs // 3),
            seed=seed + 100 + rnd
        )

        m = get_metrics_nn(model, X_test_sc, y_test)
        history.append({
            'round': rnd + 1,
            'selected': n_sel,
            'tau_h': tau_h,
            'avg_entropy_selected': float(entropy[mask].mean()),
            'test_acc': m['accuracy'],
            'test_f1': m['f1'],
            'test_auc': m['auc'],
            'test_fnr': m['fnr'],
            'test_ece': m['ece']
        })

        print(f"  Round {rnd+1:2d}: tau_H={tau_h:.3f}, selected={n_sel:3d}, "
              f"pool_left={len(X_pool):3d}, acc={m['accuracy']:.4f}, "
              f"fnr={m['fnr']:.4f}, ece={m['ece']:.4f}")

    return model, history, get_metrics_nn(model, X_test_sc, y_test)


# ---------------------------------------------------------------------------
# 5. Cost-Sensitive Asymmetric UF-SSL  (Stage 6)
# ---------------------------------------------------------------------------

def run_champion(X_lab_sc, y_labeled, X_unlab_sc, X_test_sc, y_test,
                 seed=42, tau_benign=0.95, tau_malignant=0.85,
                 h_min=0.01, h_max=0.50, class_wts=(1.0, 3.0),
                 n_epochs=50, max_rounds=10):
    """
    Safety-focused version:
    - malignant pseudo-labels are allowed with a looser confidence threshold
    - benign pseudo-labels use a stricter threshold
    - loss is class-weighted to penalize malignant mistakes more heavily
    """
    np.random.seed(seed)

    model = make_mlp(seed)
    X_train = X_lab_sc.copy()
    y_train = y_labeled.copy()
    X_pool = X_unlab_sc.copy()
    history = []

    for rnd in range(max_rounds):
        # Cost-sensitive training biases learning toward the positive/malignant class
        train_model_cost_sensitive(
            model,
            X_train,
            y_train,
            class_weights=list(class_wts),
            n_epochs=n_epochs,
            seed=seed + rnd
        )

        if len(X_pool) == 0:
            print(f"  Round {rnd+1}: unlabeled pool exhausted.")
            break

        probs = predict_proba_nn(model, X_pool)
        max_conf = probs.max(axis=1)
        pseudo_y = probs.argmax(axis=1)
        entropy = predictive_entropy(probs)

        tau_h = get_entropy_threshold(rnd, max_rounds, h_min, h_max, mode='linear')

        # Use different confidence thresholds depending on predicted class
        tau_mask = np.where(
            pseudo_y == 1,
            max_conf >= tau_malignant,
            max_conf >= tau_benign
        )

        # Final acceptance still requires low enough entropy
        mask = tau_mask & (entropy <= tau_h)
        n_sel = int(mask.sum())

        if n_sel == 0:
            print(f"  Round {rnd+1}: no samples passed dual-safety filter. Stopping early.")
            break

        X_train = np.vstack([X_train, X_pool[mask]])
        y_train = np.concatenate([y_train, pseudo_y[mask]])
        X_pool = X_pool[~mask]

        # Retrain after adding new pseudo-labels. This makes the logged metrics reflect the updated train set
        train_model_cost_sensitive(
            model,
            X_train,
            y_train,
            class_weights=list(class_wts),
            n_epochs=max(10, n_epochs // 3),
            seed=seed + 100 + rnd
        )

        m = get_metrics_nn(model, X_test_sc, y_test)
        history.append({
            'round': rnd + 1,
            'selected': n_sel,
            'tau_h': tau_h,
            'pool_left': len(X_pool),
            'test_acc': m['accuracy'],
            'test_f1': m['f1'],
            'test_auc': m['auc'],
            'test_fnr': m['fnr'],
            'test_ece': m['ece']
        })

        print(f"  Round {rnd+1:2d}: tau_H={tau_h:.3f}, selected={n_sel:3d}, "
              f"pool_left={len(X_pool):3d}, acc={m['accuracy']:.4f}, "
              f"fnr={m['fnr']:.4f}, ece={m['ece']:.4f}")

    return model, history, get_metrics_nn(model, X_test_sc, y_test)

# ---------------------------------------------------------------------------
# 6. FNR-Driven Defer Gate UF-SSL  (Stage 7)
# ---------------------------------------------------------------------------

def run_stage7(X_lab_sc, y_labeled, X_unlab_sc, X_test_sc, y_test,
               seed=42, target_fnr=None,
               tau_malignant=0.85,
               tau_benign_init=0.985, tau_benign_min=0.95, tau_benign_max=0.995,
               h_benign_init=0.04,   h_benign_min=0.02,   h_benign_max=0.15,
               tighten_step_tau=0.005, relax_step_tau=0.003,
               tighten_step_h=0.010,   relax_step_h=0.005,
               class_wts=(1.0, 3.0), n_epochs=50, max_rounds=10,
               use_stability=True, stability_k=2):
    """
    Key ideas:
    - hold out part of labeled data as a validation monitor
    - track validation FNR every round
    - tighten or relax the benign acceptance gate depending on FNR
    - if target_fnr is None, keep the benign gate fixed
    - malignant samples are handled separately
    - uncertain samples are deferred instead of forced into pseudo-labels
    """
    np.random.seed(seed)

    # Reserve a small validation set from the labeled pool to monitor FNR
    X_lab_train, X_lab_val, y_lab_train, y_lab_val = train_test_split(
        X_lab_sc, y_labeled, test_size=0.25, stratify=y_labeled, random_state=seed
    )

    model = make_mlp(seed)
    X_train = X_lab_train.copy()
    y_train = y_lab_train.copy()
    X_pool = X_unlab_sc.copy()
    history = []

    # Benign gate starts very strict and can be adjusted round by round
    tau_benign = tau_benign_init
    h_benign = h_benign_init

    # Stability tracking: how many rounds in a row each sample kept the same pseudo-label
    stable_count = np.zeros(len(X_pool), dtype=int)
    prev_pred = np.full(len(X_pool), -1, dtype=int)

    # Labeled examples start with full weight; pseudo-label weights accumulate over time
    current_weights = np.ones(len(y_train), dtype=np.float32)
    pseudo_weights_all = np.array([], dtype=np.float32)

    print(f"  Initial benign gate: tau_b={tau_benign:.3f}, H_b={h_benign:.3f}")
    print(f"  Malignant gate     : tau_m={tau_malignant:.3f}")
    if target_fnr is not None:
        print(f"  Target FNR         : {target_fnr:.4f}")
    print()

    for rnd in range(max_rounds):
        # Joint weighted training: class-sensitive + sample-sensitive
        train_model_joint_weighted(
            model,
            X_train,
            y_train,
            sample_weights_np=current_weights,
            class_weights=list(class_wts),
            n_epochs=n_epochs,
            seed=seed + rnd
        )

        # Validation FNR is used as feedback for adjusting the benign gate
        # Validation FNR is used as feedback for adjusting the benign gate
        val_metrics = get_metrics_nn(model, X_lab_val, y_lab_val)
        val_fnr = val_metrics['fnr']

        if target_fnr is None:
            # No target provided -> keep the benign gate fixed instead of
            # silently relaxing it every round
            gate_action = "hold"
        elif val_fnr > target_fnr:
            # Too many missed malignant cases -> become stricter on benign acceptance
            tau_benign = min(tau_benign + tighten_step_tau, tau_benign_max)
            h_benign = max(h_benign - tighten_step_h, h_benign_min)
            gate_action = "tighten"
        else:
            # FNR is acceptable -> cautiously relax the benign gate
            tau_benign = max(tau_benign - relax_step_tau, tau_benign_min)
            h_benign = min(h_benign + relax_step_h, h_benign_max)
            gate_action = "relax"

        if len(X_pool) == 0:
            print(f"  Round {rnd+1}: unlabeled pool exhausted.")
            break

        probs = predict_proba_nn(model, X_pool)
        pseudo_y = probs.argmax(axis=1)
        max_conf = probs.max(axis=1)
        entropy = predictive_entropy(probs)

        # Stability check: accept benign samples only if predictions stay consistent
        same_as_prev = (pseudo_y == prev_pred)
        stable_count = np.where(same_as_prev, stable_count + 1, 1)
        prev_pred = pseudo_y.copy()

        # Malignant acceptance is driven mostly by confidence on class 1
        malignant_mask = (pseudo_y == 1) & (probs[:, 1] >= tau_malignant)

        # Benign acceptance is stricter: high benign confidence + low entropy
        benign_mask = (
            (pseudo_y == 0) &
            (probs[:, 0] >= tau_benign) &
            (entropy <= h_benign)
        )

        if use_stability:
            benign_mask = benign_mask & (stable_count >= stability_k)

        # Accepted samples are pseudo-labeled; the rest are deferred for later rounds
        accept_mask = malignant_mask | benign_mask

        n_sel = int(accept_mask.sum())
        n_mal = int(malignant_mask.sum())
        n_ben = int(benign_mask.sum())
        n_def = int((~accept_mask).sum())

        if n_sel == 0:
            print(f"  Round {rnd+1}: no samples passed safety gate. Stopping early.")
            break

        # Entropy-based weights are only applied to accepted pseudo-labels
        # Benign pseudo-labels are discounted more carefully because they are riskier
        acc_entropy = entropy[accept_mask]
        acc_labels = pseudo_y[accept_mask]

        weights_sel = np.ones(n_sel, dtype=np.float32)
        ben_sel = (acc_labels == 0)

        if ben_sel.any():
            b_ent = acc_entropy[ben_sel]
            b_w = np.clip(1.0 - b_ent / max(h_benign_max, 1e-8), 0.25, 1.0)
            weights_sel[ben_sel] = b_w.astype(np.float32)

        X_train = np.vstack([X_train, X_pool[accept_mask]])
        y_train = np.concatenate([y_train, pseudo_y[accept_mask]])
        pseudo_weights_all = np.concatenate([pseudo_weights_all, weights_sel])

        # Keep only deferred samples in the pool for future rounds
        keep_pool = ~accept_mask
        X_pool = X_pool[keep_pool]
        stable_count = stable_count[keep_pool]
        prev_pred = prev_pred[keep_pool]

        # Original labeled data keeps weight 1; accepted pseudo-labels use learned weights
        current_weights = np.concatenate([
            np.ones(len(y_lab_train), dtype=np.float32),
            pseudo_weights_all.astype(np.float32)
        ])

        test_metrics = get_metrics_nn(model, X_test_sc, y_test)
        history.append({
            'round': rnd + 1,
            'val_fnr': val_fnr,
            'gate_action': gate_action,
            'tau_benign': tau_benign,
            'h_benign': h_benign,
            'selected': n_sel,
            'selected_ben': n_ben,
            'selected_mal': n_mal,
            'deferred': n_def,
            'pool_left': len(X_pool),
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
            'test_fnr': test_metrics['fnr'],
            'test_ece': test_metrics['ece'],
        })

        print(f"  Round {rnd+1:2d}: val_fnr={val_fnr:.4f} ({gate_action}), "
              f"tau_b={tau_benign:.3f}, H_b={h_benign:.3f}, "
              f"selected={n_sel:3d} [ben={n_ben:3d}, mal={n_mal:3d}], "
              f"defer={n_def:3d}, pool_left={len(X_pool):3d}, "
              f"test_fnr={test_metrics['fnr']:.4f}, ece={test_metrics['ece']:.4f}")

        # Optional short extra retrain after the newly accepted pseudo-labels are added
        train_model_joint_weighted(
            model,
            X_train,
            y_train,
            sample_weights_np=current_weights,
            class_weights=list(class_wts),
            n_epochs=max(10, n_epochs // 3),
            seed=seed + 100 + rnd
        )

    return model, history, get_metrics_nn(model, X_test_sc, y_test)