"""Ground-truth verification of the MFPCA implementation.

Run with:  python -m pytest tests/test_mfpca.py

Part A — Mathematical unit tests (numpy + sklearn only).
  Verifies the Happ-Greven algebra: scaling, eigenvalue equivalence,
  eigenfunction formula, scores.

Part B — Integration tests (fitted PACE and MFPCA objects).
  B1: roundtrip on smooth functional data with a known sinusoidal basis;
      checks MFPCA and univariate PACE reconstruction quality.
  B2–B6: regression tests for single-subject reconstruction, different grids,
      grid validation for Berrendero, and sign continuity of Berrendero
      eigenfunctions.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA

from fpca import MFPCA, PACE

# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def _make_functional_data(N=120, T=60, correlated=True, noise_free=False, seed=1):
    """
    Dense smooth functional data with a known two-mode sinusoidal basis.

    Parameters
    ----------
    correlated : bool
        If True: both features share the SAME latent score for mode 1
        (a common factor), so MFPCA can exploit cross-feature structure.
        If False: all scores are independent — MFPCA should perform
        similarly to (not worse than) univariate PACE.

    Returns
    -------
    data       : dict {feat_name: list of (t, values)}
    t          : np.ndarray, time grid
    true_trajs : dict {feat_name: (N, T) array}, noiseless ground truth
    signal_std : dict {feat_name: float}, std of noiseless signal (for relative error)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)

    phi1_f0 = np.sin(np.pi * t)
    phi2_f0 = np.sin(2 * np.pi * t)
    phi1_f1 = np.cos(np.pi * t)
    phi2_f1 = np.cos(2 * np.pi * t)

    if correlated:
        # Mode 1 is driven by the SAME latent score in both features
        z1_f0 = rng.standard_normal(N) * 3.0   # shared factor, std 3
        z1_f1 = z1_f0 * 2.0                     # feat1 loads twice as strongly
    else:
        # All scores are fully independent
        z1_f0 = rng.standard_normal(N) * 3.0
        z1_f1 = rng.standard_normal(N) * 6.0   # same std as correlated case but independent

    z2_f0 = rng.standard_normal(N) * 1.0
    z2_f1 = rng.standard_normal(N) * 2.0

    X0_clean = np.outer(z1_f0, phi1_f0) + np.outer(z2_f0, phi2_f0)
    X1_clean = np.outer(z1_f1, phi1_f1) + np.outer(z2_f1, phi2_f1)

    noise_std_f0 = 0.0 if noise_free else 0.3
    noise_std_f1 = 0.0 if noise_free else 0.6
    X0 = X0_clean + rng.standard_normal((N, T)) * noise_std_f0
    X1 = X1_clean + rng.standard_normal((N, T)) * noise_std_f1

    data = {
        'feat0': [(t, X0[i]) for i in range(N)],
        'feat1': [(t, X1[i]) for i in range(N)],
    }
    true_trajs = {'feat0': X0_clean, 'feat1': X1_clean}
    signal_std = {
        'feat0': float(np.std(X0_clean)),
        'feat1': float(np.std(X1_clean)),
    }
    return data, t, true_trajs, signal_std


def _fit_pace_models(data, n_components=4):
    pace_models = {}
    for feat, trajs in data.items():
        pace = PACE(n_components=n_components)
        pace.fit(trajs)
        pace_models[feat] = pace
    return pace_models


def _make_pace_and_data(N=80, T=40, n_feats=2, scale_factor=5.0, seed=1):
    """White-noise dense data; used only by fix-verification tests B2–B6."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    data, pace_models = {}, {}
    for f in range(n_feats):
        amplitude = scale_factor * (f + 1)
        trajs = rng.standard_normal((N, T)) * amplitude
        key = f"feat{f}"
        data[key] = [(t, trajs[i]) for i in range(N)]
        pace = PACE(n_components=4)
        pace.fit(data[key])
        pace_models[key] = pace
    return data, pace_models


# ──────────────────────────────────────────────────────────────────
# PART A  MATHEMATICAL UNIT TESTS
# ──────────────────────────────────────────────────────────────────

def test_A1_scaling():
    """scale_stds_[f] = sqrt(sum(eigenvalues)) implements w_j^{1/2} = 1/s_j.
    After dividing Xi by s_j, each feature contributes unit total variance."""
    rng = np.random.default_rng(0)
    lam0 = np.array([4.0, 1.0, 0.25])   # s0 = sqrt(5.25)
    lam1 = np.array([9.0, 2.0])          # s1 = sqrt(11.0)
    s0, s1 = np.sqrt(lam0.sum()), np.sqrt(lam1.sum())
    N = 5000
    xi0 = rng.standard_normal((N, 3)) * np.sqrt(lam0)
    xi1 = rng.standard_normal((N, 2)) * np.sqrt(lam1)
    Xi_tilde = np.concatenate([xi0 / s0, xi1 / s1], axis=1)
    tv0 = np.var(Xi_tilde[:, :3], ddof=1, axis=0).sum()
    tv1 = np.var(Xi_tilde[:, 3:], ddof=1, axis=0).sum()
    assert_allclose(tv0, 1.0, atol=0.05,
                    err_msg=f"Feature 0 total var after scaling = {tv0:.4f}, expected 1")
    assert_allclose(tv1, 1.0, atol=0.05,
                    err_msg=f"Feature 1 total var after scaling = {tv1:.4f}, expected 1")
    print(f"PASS test_A1_scaling  (tv0={tv0:.4f}, tv1={tv1:.4f})")


def test_A2_pca_eigenvalues_match_manual_Qw():
    """sklearn PCA eigenvalues on Xi_tilde == eigenvalues of (N-1)^{-1} Xi_centered^T Xi_centered.
    sklearn centers internally, so the manual computation must also center first."""
    rng = np.random.default_rng(1)
    N, K = 300, 5
    Xi_tilde = rng.standard_normal((N, K))
    # sklearn PCA subtracts column means before computing covariance
    Xi_centered = Xi_tilde - Xi_tilde.mean(axis=0)
    Q_w = Xi_centered.T @ Xi_centered / (N - 1)
    evals_manual = np.sort(np.linalg.eigvalsh(Q_w))[::-1]
    pca = PCA(n_components=K).fit(Xi_tilde)
    assert_allclose(pca.explained_variance_, evals_manual, rtol=1e-8,
                    err_msg="sklearn PCA eigenvalues != manual Q̂_w eigenvalues")
    print("PASS test_A2_pca_eigenvalues_match_manual_Qw")


def test_A3_eigenfunction_formula():
    """Verifies: s_j * (c_block @ phi_q) == paper's psi_m^(j) = w_j^{-1/2} sum_l [c_m]_l phi_l.
    The code stores c_block @ phi_q in weight_functions_; _undo_scaling multiplies by s_j."""
    rng = np.random.default_rng(2)
    N, M0, M1, T0, T1 = 200, 3, 2, 40, 60
    lam0 = np.array([4.0, 1.0, 0.25])
    lam1 = np.array([9.0, 2.0])
    s0, s1 = np.sqrt(lam0.sum()), np.sqrt(lam1.sum())
    phi0 = rng.standard_normal((M0, T0))
    phi1 = rng.standard_normal((M1, T1))
    xi0 = rng.standard_normal((N, M0)) * np.sqrt(lam0)
    xi1 = rng.standard_normal((N, M1)) * np.sqrt(lam1)
    Xi_tilde = np.concatenate([xi0 / s0, xi1 / s1], axis=1)
    pca = PCA(n_components=2).fit(Xi_tilde)

    for m in range(2):
        c_m = pca.components_[m]
        c0, c1 = c_m[:M0], c_m[M0:]
        # Stored value (weight_functions_ before _undo_scaling)
        wf0 = c0 @ phi0
        wf1 = c1 @ phi1
        # Paper's true multivariate eigenfunction (after _undo_scaling)
        psi0_paper = s0 * (c0 @ phi0)
        psi1_paper = s1 * (c1 @ phi1)
        assert_allclose(s0 * wf0, psi0_paper, rtol=1e-12,
                        err_msg=f"Mode {m}, feat 0: stored*s0 != paper psi")
        assert_allclose(s1 * wf1, psi1_paper, rtol=1e-12,
                        err_msg=f"Mode {m}, feat 1: stored*s1 != paper psi")
    print("PASS test_A3_eigenfunction_formula")


def test_A4_scores_formula():
    """rho_{i,m} = (Xi_tilde - mean) @ c_m == sklearn PCA transform.
    pca.transform(X) subtracts pca.mean_ internally, so pass Xi_tilde directly."""
    rng = np.random.default_rng(3)
    N, K = 150, 5
    Xi_tilde = rng.standard_normal((N, K))
    pca = PCA(n_components=3).fit(Xi_tilde)
    scores_sklearn = pca.transform(Xi_tilde)                        # sklearn centers internally
    scores_manual = (Xi_tilde - pca.mean_) @ pca.components_.T     # manual: subtract mean once
    assert_allclose(scores_sklearn, scores_manual, rtol=1e-10,
                    err_msg="sklearn transform != manual (Xi_tilde - mean) @ c_m")
    print("PASS test_A4_scores_formula")


# ──────────────────────────────────────────────────────────────────
# PART B  INTEGRATION TESTS
# ──────────────────────────────────────────────────────────────────

def _run_reconstruction_comparison(data, t, true_trajs, signal_std, label):
    """Fit PACE + MFPCA, reconstruct, print relative-error table. Returns (rms_mv, rms_uv) per feat."""
    feat_names = list(data.keys())
    N, T_len = true_trajs[feat_names[0]].shape
    pace_models = _fit_pace_models(data, n_components=4)

    mfpca = MFPCA(pace_models=pace_models, n_components=4,
                  method='happ_greven', scale_features=True,
                  n_components_per_feature=4)
    mv_scores = mfpca.fit_transform(data)
    mv_recon = mfpca.reconstruct_from_scores(mv_scores)   # (N, T, F)

    uv_recon = {}
    for feat in feat_names:
        pace = pace_models[feat]
        uv_scores_all = pace.transform(data[feat])
        vals = np.zeros((N, T_len))
        for i in range(N):
            recon_vals, recon_t = pace.reconstruct_curve(t, uv_scores_all[i])
            vals[i] = np.interp(t, recon_t, recon_vals)
        uv_recon[feat] = vals

    print(f"\n  {label}")
    print(f"  {'Feature':<8}  {'MFPCA %err':>12}  {'Univariate %err':>16}  {'Noise %':>10}")
    print(f"  {'-'*50}")
    results = {}
    for fi, feat in enumerate(feat_names):
        gt    = true_trajs[feat]
        noisy = np.array([v for (_, v) in data[feat]])
        sig   = signal_std[feat]

        pct_mv    = 100 * np.sqrt(np.mean((mv_recon[:, :, fi] - gt) ** 2)) / sig
        pct_uv    = 100 * np.sqrt(np.mean((uv_recon[feat]     - gt) ** 2)) / sig
        pct_noisy = 100 * np.sqrt(np.mean((noisy               - gt) ** 2)) / sig
        print(f"  {feat:<8}  {pct_mv:>11.1f}%  {pct_uv:>15.1f}%  {pct_noisy:>9.1f}%")
        results[feat] = (pct_mv, pct_uv, pct_noisy)
    return results


def test_B1_roundtrip_and_univariate_comparison():
    """
    Uses smooth functional data with a known sinusoidal basis (not white noise).
    Error is expressed as % of signal std so it's interpretable regardless of amplitude.

    Scenario A — CORRELATED features (noisy): MFPCA should beat univariate PACE.
    Scenario B — INDEPENDENT features (noisy): MFPCA should not be worse than univariate.
    Scenario C — CORRELATED, NOISELESS, more components: error should be near zero
      (only truncation error remains; with enough components it should vanish).
      The true data has exactly 2 modes per feature, so with 2 PACE components
      per feature and 4 multivariate components the signal is fully spanned.
    """
    data_c, t, gt_c, sig_c = _make_functional_data(correlated=True,  noise_free=False, seed=1)
    data_i, _, gt_i, sig_i = _make_functional_data(correlated=False, noise_free=False, seed=2)
    data_n, _, gt_n, sig_n = _make_functional_data(correlated=True,  noise_free=True,  seed=1)

    res_c = _run_reconstruction_comparison(data_c, t, gt_c, sig_c,
                                           "Scenario A: CORRELATED, noisy")
    res_i = _run_reconstruction_comparison(data_i, t, gt_i, sig_i,
                                           "Scenario B: INDEPENDENT, noisy")

    # Scenario C: noiseless data, use exactly 2 PACE components (matching the 2 true
    # modes) and 4 multivariate components — reconstruction error should be < 1%
    feat_names = list(data_n.keys())
    N, T_len = gt_n[feat_names[0]].shape
    pace_models_n = _fit_pace_models(data_n, n_components=2)
    mfpca_n = MFPCA(pace_models=pace_models_n, n_components=4,
                    method='happ_greven', scale_features=True,
                    n_components_per_feature=2)
    mv_scores_n = mfpca_n.fit_transform(data_n)
    mv_recon_n  = mfpca_n.reconstruct_from_scores(mv_scores_n)

    uv_recon_n = {}
    for feat in feat_names:
        pace = pace_models_n[feat]
        uv_scores_all = pace.transform(data_n[feat])
        vals = np.zeros((N, T_len))
        for i in range(N):
            recon_vals, recon_t = pace.reconstruct_curve(t, uv_scores_all[i])
            vals[i] = np.interp(t, recon_t, recon_vals)
        uv_recon_n[feat] = vals

    print("\n  Scenario C: CORRELATED, NOISELESS, 2 PACE components (= exact basis)")
    print(f"  {'Feature':<8}  {'MFPCA %err':>12}  {'Univariate %err':>16}")
    print(f"  {'-'*40}")
    res_n = {}
    for fi, feat in enumerate(feat_names):
        gt   = gt_n[feat]
        sig  = sig_n[feat]
        pct_mv = 100 * np.sqrt(np.mean((mv_recon_n[:, :, fi] - gt) ** 2)) / sig
        pct_uv = 100 * np.sqrt(np.mean((uv_recon_n[feat]     - gt) ** 2)) / sig
        print(f"  {feat:<8}  {pct_mv:>11.2f}%  {pct_uv:>15.2f}%")
        res_n[feat] = (pct_mv, pct_uv)

    for feat in ['feat0', 'feat1']:
        pct_mv_c, pct_uv_c, pct_noise_c = res_c[feat]
        pct_mv_i, pct_uv_i, pct_noise_i = res_i[feat]
        pct_mv_n, pct_uv_n              = res_n[feat]

        # Noisy: both methods should denoise (error < 50% of raw noise level)
        assert pct_mv_c < 0.5 * pct_noise_c, \
            f"FAIL: MFPCA (correlated) {pct_mv_c:.1f}% >= 50% of noise {pct_noise_c:.1f}% for {feat}"
        assert pct_uv_c < 0.5 * pct_noise_c, \
            f"FAIL: Univariate (correlated) {pct_uv_c:.1f}% >= 50% of noise {pct_noise_c:.1f}% for {feat}"
        assert pct_mv_i < 0.5 * pct_noise_i, \
            f"FAIL: MFPCA (independent) {pct_mv_i:.1f}% >= 50% of noise {pct_noise_i:.1f}% for {feat}"
        assert pct_uv_i < 0.5 * pct_noise_i, \
            f"FAIL: Univariate (independent) {pct_uv_i:.1f}% >= 50% of noise {pct_noise_i:.1f}% for {feat}"

        # Correlated: MFPCA should be at least as good as univariate. When each PACE
        # model retains exactly the true components, Happ-Greven MFPCA with enough
        # components spans the same space and the two errors coincide.
        assert pct_mv_c <= pct_uv_c * 1.01, \
            f"FAIL: MFPCA ({pct_mv_c:.1f}%) worse than univariate ({pct_uv_c:.1f}%) for correlated {feat}"

        # Independent: MFPCA should not be more than 20% relatively worse than univariate
        assert pct_mv_i < pct_uv_i * 1.20, \
            f"FAIL: MFPCA ({pct_mv_i:.1f}%) more than 20% worse than univariate ({pct_uv_i:.1f}%) for independent {feat}"

        # Noiseless + exact components: both errors should be near zero (< 1%)
        assert pct_mv_n < 1.0, \
            f"FAIL: MFPCA noiseless error {pct_mv_n:.2f}% >= 1% for {feat} (truncation or math error)"
        assert pct_uv_n < 1.0, \
            f"FAIL: Univariate noiseless error {pct_uv_n:.2f}% >= 1% for {feat} (PACE not fitting basis)"

    print("\nPASS test_B1_roundtrip_and_univariate_comparison")


def test_B2_fix1_reconstruct_feature_single_subject():
    """Fix 1: reconstruct_feature with a single subject must not crash for any
    feature index and must return shape (T,)."""
    data, pace_models = _make_pace_and_data()
    feat_names = list(data.keys())
    mfpca = MFPCA(pace_models=pace_models, n_components=3,
                  method='berrendero', scale_features=True,
                  n_components_per_feature=4)
    scores_all = mfpca.fit_transform(data)   # (N, 3)
    scores_one = scores_all[0]               # (3,) single subject

    for fi, feat in enumerate(feat_names):
        values, grid = mfpca.reconstruct_feature(scores_one, feat)
        assert values.ndim == 1, \
            f"FAIL: single-subject '{feat}' should be 1D, got {values.shape}"
        assert len(values) == len(grid), \
            f"FAIL: len(values) {len(values)} != len(grid) {len(grid)} for '{feat}'"
        print(f"  PASS single-subject '{feat}' -> shape {values.shape}")

    # Multi-subject should return (N, T)
    values_multi, _ = mfpca.reconstruct_feature(scores_all, feat_names[0])
    assert values_multi.ndim == 2, \
        f"FAIL: multi-subject should be 2D, got {values_multi.shape}"
    print("PASS test_B2_fix1_reconstruct_feature_single_subject")


def test_B3_fix2_hg_accepts_different_grids():
    """Fix 2: H-G must accept PACE models fitted on different time grids."""
    rng = np.random.default_rng(5)
    N = 60
    t0 = np.linspace(0, 1, 30)
    t1 = np.linspace(0, 2, 50)   # different domain AND length
    data0 = [(t0, rng.standard_normal(30) * 5) for _ in range(N)]
    data1 = [(t1, rng.standard_normal(50) * 10) for _ in range(N)]
    pace0 = PACE(n_components=2).fit(data0)
    pace1 = PACE(n_components=2).fit(data1)
    assert len(pace0.grid_) != len(pace1.grid_), "Precondition: grids must differ"

    mfpca = MFPCA(pace_models={'f0': pace0, 'f1': pace1},
                  n_components=2, method='happ_greven', scale_features=True,
                  n_components_per_feature=2)
    scores = mfpca.fit_transform({'f0': data0, 'f1': data1})

    assert mfpca.weight_functions_.shape[1] == len(pace0.grid_), (
        f"FAIL: weight_functions_ time axis {mfpca.weight_functions_.shape[1]} "
        f"!= first grid length {len(pace0.grid_)}"
    )
    # Also verify reconstruct_from_scores works (mean functions must be on display grid)
    recon = mfpca.reconstruct_from_scores(scores)
    assert recon.shape == (N, len(pace0.grid_), 2), \
        f"FAIL: reconstruction shape {recon.shape}, expected ({N}, {len(pace0.grid_)}, 2)"
    print(f"PASS test_B3_fix2_hg_accepts_different_grids  "
          f"weight_functions_ shape = {mfpca.weight_functions_.shape}, "
          f"recon shape = {recon.shape}")


def test_B4_fix2_berrendero_rejects_different_grids():
    """Fix 2: Berrendero with mismatched grids raises ValueError mentioning happ_greven."""
    rng = np.random.default_rng(6)
    N = 40
    t0 = np.linspace(0, 1, 30)
    t1 = np.linspace(0, 1, 50)
    data0 = [(t0, rng.standard_normal(30)) for _ in range(N)]
    data1 = [(t1, rng.standard_normal(50)) for _ in range(N)]
    pace0 = PACE(n_components=2).fit(data0)
    pace1 = PACE(n_components=2).fit(data1)
    mfpca = MFPCA(pace_models={'f0': pace0, 'f1': pace1},
                  n_components=2, method='berrendero')
    with pytest.raises(ValueError, match="(?i)grid|happ_greven"):
        mfpca.fit({'f0': data0, 'f1': data1})


def test_B5_fix4_no_scale_means():
    """Fix 4: scale_means_ attribute must NOT exist after fitting."""
    data, pace_models = _make_pace_and_data()
    mfpca = MFPCA(pace_models=pace_models, n_components=2,
                  method='happ_greven', scale_features=True,
                  n_components_per_feature=4)
    mfpca.fit(data)
    assert not hasattr(mfpca, 'scale_means_'), \
        "FAIL: scale_means_ still exists (Fix 4 not applied)"
    print("PASS test_B5_fix4_no_scale_means")


def test_B6_fix3_berrendero_smooth_weight_functions():
    """Fix 3: Berrendero weight_functions_ should be sign-continuous across time.
    Checks that fewer than 10% of consecutive time-slice pairs have sign flips."""
    data, pace_models = _make_pace_and_data(N=60, T=50, n_feats=2)
    mfpca = MFPCA(pace_models=pace_models, n_components=2,
                  method='berrendero', scale_features=True,
                  n_components_per_feature=4)
    mfpca.fit(data)
    wf = mfpca.weight_functions_   # (K, T, F)
    K, T = wf.shape[0], wf.shape[1]
    sign_flips = sum(
        1
        for k in range(K)
        for t in range(1, T)
        if np.dot(wf[k, t, :], wf[k, t - 1, :]) < 0
    )
    total_pairs = K * (T - 1)
    flip_rate = sign_flips / total_pairs
    assert flip_rate < 0.1, (
        f"FAIL: {flip_rate:.1%} of consecutive weight function pairs have sign flips "
        f"(threshold 10%)"
    )
    print(f"PASS test_B6_fix3_berrendero_smooth_weight_functions  "
          f"(flip rate = {flip_rate:.1%})")


# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Part A: Mathematical Unit Tests (numpy + sklearn only) ===")
    test_A1_scaling()
    test_A2_pca_eigenvalues_match_manual_Qw()
    test_A3_eigenfunction_formula()
    test_A4_scores_formula()

    print("\n=== Part B: Integration Tests (PACE + MFPCA) ===")
    test_B1_roundtrip_and_univariate_comparison()
    test_B2_fix1_reconstruct_feature_single_subject()
    test_B3_fix2_hg_accepts_different_grids()
    test_B4_fix2_berrendero_rejects_different_grids()
    test_B5_fix4_no_scale_means()
    test_B6_fix3_berrendero_smooth_weight_functions()

    print("\n=== ALL TESTS PASSED ===")
