#include "optimizers/Marginalization.hpp"

#include <mutex>
#include <thread>

namespace orion_slam {

// ─────────────────────────────────────────────────────────────────────────────
// MarginalizationBlockInfo::Evaluate
//
// Linearises the wrapped Ceres factor at the current parameter values.
// Allocates _raw_jacobians (an array of pointers into the Eigen matrices);
// the caller is responsible for freeing it with delete[] after use.
// ─────────────────────────────────────────────────────────────────────────────
void MarginalizationBlockInfo::Evaluate() {

    _residuals.resize(_cost_function->num_residuals());

    const std::vector<int>& block_sizes = _cost_function->parameter_block_sizes();
    _raw_jacobians = new double*[block_sizes.size()];
    _jacobians.resize(block_sizes.size());

    for (size_t i = 0; i < block_sizes.size(); i++) {
        _jacobians[i].resize(_cost_function->num_residuals(), block_sizes[i]);
        _raw_jacobians[i] = _jacobians[i].data();
    }

    _cost_function->Evaluate(_parameter_blocks.data(), _residuals.data(), _raw_jacobians);
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::preMarginalize
//
// Classifies all variables connected to frame0 into three groups:
//
//   Marginalise (_m DOF)
//     • frame0 pose (6) + IMU states (9) if VIO
//     • landmarks seen ONLY from frame0  ("lonely" landmarks)
//
//   Keep (_n DOF)
//     • frame1 pose (6) + IMU states (9) if VIO  ← always, not just in VIO
//     • landmarks also seen from other remaining frames
//
//   Ignore
//     • outliers, uninitialised landmarks, landmarks without enough constraints
//
// The column layout of the full (m+n)×(m+n) information matrix is:
//   [frame0 | lonely_lmks | frame1 | shared_lmks | extra_lmks_from_prior]
//    ←──────── _m ────────→ ←──────────── _n ──────────────────────────→
//
// marginalization_last carries landmarks that survived from the previous step
// (e.g. re-observed landmarks whose prior must be propagated forward).
// ─────────────────────────────────────────────────────────────────────────────
void Marginalization::preMarginalize(std::shared_ptr<Frame>&          frame0,
                                     std::shared_ptr<Frame>&          frame1,
                                     std::shared_ptr<Marginalization>& marginalization_last) {
    // ── Reset state ────────────────────────────────────────────────────────────
    _frame_to_marg = frame0;
    _frame_to_keep = nullptr;
    _lmk_to_keep.clear();
    _lmk_to_marg.clear();
    _map_frame_idx.clear();
    _map_lmk_idx.clear();
    _map_lmk_inf.clear();
    _map_lmk_prior.clear();
    _m = 0;
    _n = 0;

    // ── Variables to MARGINALISE ───────────────────────────────────────────────

    // frame0 pose is always marginalised (6 DOF).
    _map_frame_idx.emplace(frame0, 0);
    _m = 6;
    int next_idx = 6;

    // VIO: also marginalise frame0's velocity + IMU biases (+9 DOF).
    if (frame0->getIMU()) {
        _m       += 9;
        next_idx += 9;
    }

    // Classify each landmark linked to frame0.
    for (const auto& [type, lmks] : frame0->getLandmarks()) {
        const int lmk_dof = (type == "pointxd") ? 3 : 6;

        for (const auto& lmk : lmks) {
            if (lmk->isOutlier() || !lmk->isInMap() || !lmk->isInitialized())
                continue;

            // Count how many cameras (other than frame0) observe this landmark,
            // and whether frame0 itself observes it from both stereo cameras.
            bool is_lonely  = true;
            int  num_cam_f0 = 0;

            for (const auto& weak_f : lmk->getFeatures()) {
                auto feat = weak_f.lock();
                if (!feat) continue;
                if (feat->getSensor()->getFrame() != frame0)
                    is_lonely = false;    // another frame sees this landmark
                else
                    num_cam_f0++;
            }

            // Landmarks without a stereo observation from frame0 AND without a
            // prior cannot be triangulated here — skip them.
            if (num_cam_f0 < 2 && !lmk->hasPrior()) {
                lmk->setMarg();
                continue;
            }

            if (is_lonely) {
                // Seen only from frame0 → marginalise together with frame0.
                _lmk_to_marg[type].push_back(lmk);
                lmk->setMarg();
                _map_lmk_idx.emplace(lmk, next_idx);
                _m       += lmk_dof;
                next_idx += lmk_dof;
            } else {
                // Seen from other frames too → keep and give a prior.
                _lmk_to_keep[type].push_back(lmk);
                lmk->setPrior();
                _n += lmk_dof;
                // Index assigned after frame1 block (see below).
            }
        }
    }

    // ── Variables to KEEP ─────────────────────────────────────────────────────

    // frame1 is ALWAYS added to the kept variables (VO and VIO alike).
    // This ensures the Schur complement creates a prior on frame1's pose,
    // so the prior chain does not break when IMU is unavailable.
    _frame_to_keep = frame1;
    _map_frame_idx.emplace(frame1, next_idx);
    const int frame1_dof = frame1->getIMU() ? 15 : 6;
    _n       += frame1_dof;
    next_idx += frame1_dof;

    // Assign column indices for the shared landmarks (to keep).
    for (const auto& [type, lmks] : _lmk_to_keep) {
        const int lmk_dof = (type == "pointxd") ? 3 : 6;
        for (const auto& lmk : lmks) {
            _map_lmk_idx.emplace(lmk, next_idx);
            next_idx += lmk_dof;
        }
    }

    // ── Carry over landmarks from the previous prior ───────────────────────────
    // If a landmark survived the last marginalisation but is not linked to the
    // current frame, it still needs to appear in the kept variable set so that
    // the prior information is not lost (resurrection case).
    if (marginalization_last->_lmk_to_keep.empty())
        return;

    bool discard_prior = false;

    for (const auto& [type, lmks] : marginalization_last->_lmk_to_keep) {
        const int lmk_dof = (type == "pointxd") ? 3 : 6;
        for (const auto& lmk : lmks) {

            if (_map_lmk_idx.count(lmk))
                continue;  // already registered above

            if (lmk->isOutlier()) {
                // An outlier in the prior invalidates the whole prior
                // (the linearisation point is no longer valid).
                discard_prior = true;
                break;
            }

            _lmk_to_keep[type].push_back(lmk);
            _map_lmk_idx.emplace(lmk, next_idx);
            _n       += lmk_dof;
            next_idx += lmk_dof;
        }
        if (discard_prior) break;
    }

    if (discard_prior)
        marginalization_last->_lmk_to_keep.clear();
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::computeInformationAndGradient
//
// Accumulates the normal-equation contributions from all registered factors:
//   A += Σ  Jᵢᵀ Jᵢ
//   b += Σ  Jᵢᵀ rᵢ
//
// Each block is evaluated (linearised) here.  The work is split across
// 4 threads; a mutex protects the shared A and b.
// ─────────────────────────────────────────────────────────────────────────────
void Marginalization::computeInformationAndGradient(
    std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
    Eigen::MatrixXd& A,
    Eigen::VectorXd& b) {

    std::mutex mtx;

    auto accumulate = [&](std::vector<std::shared_ptr<MarginalizationBlockInfo>> chunk) {
        for (auto& block : chunk) {
            block->Evaluate();

            const auto& block_sizes = block->_cost_function->parameter_block_sizes();

            for (size_t i = 0; i < block_sizes.size(); i++) {
                const int idx_i  = block->_parameter_idx[i];
                const int size_i = block_sizes[i];
                if (idx_i == -1) continue;

                const Eigen::MatrixXd& Ji = block->_jacobians[i];

                for (size_t j = i; j < block_sizes.size(); j++) {
                    const int idx_j  = block->_parameter_idx[j];
                    const int size_j = block_sizes[j];
                    if (idx_j == -1) continue;

                    std::lock_guard<std::mutex> lock(mtx);
                    Eigen::MatrixXd JiTJj = Ji.transpose() * block->_jacobians[j];
                    A.block(idx_i, idx_j, size_i, size_j) += JiTJj;
                    if (i != j)
                        A.block(idx_j, idx_i, size_j, size_i) += JiTJj.transpose();
                }

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    b.segment(idx_i, size_i) += Ji.transpose() * block->_residuals;
                }
            }
        }
    };

    // Distribute blocks across 4 threads.
    constexpr int N_THREADS = 4;
    std::vector<std::vector<std::shared_ptr<MarginalizationBlockInfo>>> chunks(N_THREADS);
    for (size_t i = 0; i < blocks.size(); i++)
        chunks[i % N_THREADS].push_back(blocks[i]);

    std::vector<std::thread> threads;
    for (auto& chunk : chunks)
        threads.emplace_back(accumulate, chunk);
    for (auto& t : threads)
        t.join();
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::computeSchurComplement
//
// 1. Build the full (m+n)×(m+n) information matrix A and gradient b.
// 2. Extract the four blocks:
//      Amm (m×m)  Amr (m×n)
//      Arm (n×m)  Arr (n×n)
// 3. Invert Amm robustly (via eigendecomposition, ignoring near-zero modes).
// 4. Schur complement:
//      Ak = Arr − Arm · Amm⁻¹ · Armᵀ      (prior information on kept vars)
//      bk = brr − Arm · Amm⁻¹ · bmm      (prior gradient  on kept vars)
// 5. Shift the variable indices from the full (m+n) space to the kept (n) space.
// 6. Apply rank-revealing decomposition on Ak.
// ─────────────────────────────────────────────────────────────────────────────
bool Marginalization::computeSchurComplement() {

    if (_n < 4)
        return false;

    // Build the full information matrix.
    const int total = _m + _n;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(total, total);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(total);
    computeInformationAndGradient(_marginalization_blocks, A, b);

    // Free the temporary Jacobian pointer arrays allocated by Evaluate().
    // Note: _cost_function ownership belongs to the caller (typically Ceres),
    // so we do NOT delete it here.
    for (auto& block : _marginalization_blocks) {
        delete[] block->_raw_jacobians;
        block->_raw_jacobians = nullptr;
    }
    _marginalization_blocks.clear();

    // ── Schur complement ───────────────────────────────────────────────────────
    // Symmetrise Amm (numerical noise may break exact symmetry).
    Eigen::MatrixXd Amm = 0.5 * (A.topLeftCorner(_m, _m) +
                                  A.topLeftCorner(_m, _m).transpose());

    // Robust pseudo-inverse of Amm via eigendecomposition.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv =
        saes.eigenvectors() *
        Eigen::VectorXd((saes.eigenvalues().array() > _eps)
                            .select(saes.eigenvalues().array().inverse(), 0.0))
            .asDiagonal() *
        saes.eigenvectors().transpose();

    const Eigen::VectorXd bmm = b.head(_m);
    const Eigen::MatrixXd Arm = A.block(_m, 0,  _n, _m);
    const Eigen::MatrixXd Arr = A.block(_m, _m, _n, _n);
    const Eigen::VectorXd brr = b.tail(_n);

    _Ak = Arr - Arm * Amm_inv * Arm.transpose();
    _bk = brr - Arm * Amm_inv * bmm;

    // ── Shift indices to the kept subspace [0, n) ─────────────────────────────
    for (auto& [lmk, idx] : _map_lmk_idx)
        idx -= _m;

    if (_frame_to_keep)
        _map_frame_idx.at(_frame_to_keep) -= _m;

    // ── Rank-revealing decomposition of Ak ────────────────────────────────────
    rankReveallingDecomposition(_Ak, _U, _Lambda);
    _Sigma   = _Lambda.array().inverse();
    _n_full  = _U.cols();
    _Sigma_k = _U * _Sigma.asDiagonal() * _U.transpose();

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::computeJacobiansAndResiduals
//
// Converts the information-form prior (Ak, bk) into the Jacobian / residual
// form expected by Ceres inside MarginalizationFactor:
//
//   Ak = Uᵀ Λ U = Jᵀ J    →    J  = Λ^{½} Uᵀ
//   Gauss-Newton: Ak δx = −bk  →   Jᵀ r₀ = −bk
//                                    r₀ = −Λ^{−½} Uᵀ bk
// ─────────────────────────────────────────────────────────────────────────────
bool Marginalization::computeJacobiansAndResiduals() {

    _marginalization_jacobian = _Lambda.cwiseSqrt().asDiagonal() * _U.transpose();
    _marginalization_residual = -(_Sigma.cwiseSqrt().asDiagonal() * _U.transpose() * _bk);

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::rankReveallingDecomposition
//
// Thin eigendecomposition of A: only keeps eigenvectors whose eigenvalue
// exceeds _eps.  Result stored in U (n × rank) and d (rank).
// ─────────────────────────────────────────────────────────────────────────────
void Marginalization::rankReveallingDecomposition(Eigen::MatrixXd  A,
                                                   Eigen::MatrixXd& U,
                                                   Eigen::VectorXd& d) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(A);

    const Eigen::VectorXd& evals = saes.eigenvalues();
    const int n     = evals.size();
    const int rank  = (evals.array() > _eps).count();

    U = Eigen::MatrixXd::Zero(n, rank);
    d = Eigen::VectorXd::Zero(rank);

    int k = 0;
    for (int i = 0; i < n; i++) {
        if (evals(i) > _eps) {
            U.col(k) = saes.eigenvectors().col(i);
            d(k)     = evals(i);
            k++;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Information metrics (used by sparsifyVO)
// ─────────────────────────────────────────────────────────────────────────────

double Marginalization::computeEntropy(std::shared_ptr<ALandmark> lmk) {
    const int size = (lmk->getLandmarkLabel() == "pointxd") ? 3 : 6;
    const Eigen::MatrixXd& Sig = _Sigma_k.block(_map_lmk_idx.at(lmk),
                                                 _map_lmk_idx.at(lmk),
                                                 size, size);
    return std::log(std::pow(2 * M_PI * M_E, size / 2.0) * Sig.determinant());
}

double Marginalization::computeMutualInformation(std::shared_ptr<ALandmark> lmk_i,
                                                  std::shared_ptr<ALandmark> lmk_j) {
    const int si = (lmk_i->getLandmarkLabel() == "pointxd") ? 3 : 6;
    const int sj = (lmk_j->getLandmarkLabel() == "pointxd") ? 3 : 6;
    const int ii = _map_lmk_idx.at(lmk_i);
    const int ij = _map_lmk_idx.at(lmk_j);

    const Eigen::MatrixXd Sii = _Sigma_k.block(ii, ii, si, si);
    const Eigen::MatrixXd Sij = _Sigma_k.block(ii, ij, si, sj);
    const Eigen::MatrixXd Sjj = _Sigma_k.block(ij, ij, sj, sj);

    Eigen::MatrixXd S(si + sj, si + sj);
    S.topLeftCorner(si, si)     = Sii;
    S.topRightCorner(si, sj)    = Sij;
    S.bottomLeftCorner(sj, si)  = Sij.transpose();
    S.bottomRightCorner(sj, sj) = Sjj;

    // MI = ½ log( det(Sii)·det(Sjj) / det(S) )
    return 0.5 * std::log(Sii.determinant() * Sjj.determinant() / S.determinant());
}

double Marginalization::computeOffDiag(std::shared_ptr<ALandmark> lmk_i,
                                        std::shared_ptr<ALandmark> lmk_j) {
    const int si = (lmk_i->getLandmarkLabel() == "pointxd") ? 3 : 6;
    const int sj = (lmk_j->getLandmarkLabel() == "pointxd") ? 3 : 6;
    return std::abs(_Ak.block(_map_lmk_idx.at(lmk_i),
                               _map_lmk_idx.at(lmk_j),
                               si, sj).trace());
}

double Marginalization::computeKLD(Eigen::MatrixXd A_p, Eigen::MatrixXd A_q) {
    Eigen::MatrixXd U;
    Eigen::VectorXd d;
    rankReveallingDecomposition(A_p, U, d);

    const Eigen::VectorXd Sigma_p = d.array().inverse();
    const Eigen::MatrixXd delta   = U.transpose() * A_q * U * Sigma_p.asDiagonal();
    const double det = delta.determinant();

    if (det == 0.0)
        return 100.0;  // degenerate case

    // KL(p||q) = ½ (tr(Σ_p A_q) − log det(Σ_p A_q) − k)
    return 0.5 * (delta.trace() - std::log(det) - U.cols());
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::sparsifyVIO
//
// Replaces the dense (_n_full × _n) prior with a sparse set of factors:
//   • one absolute factor per kept landmark (information ≈ marginal of Ak)
//   • one absolute frame factor for frame_to_keep
//
// Each factor's information matrix is approximated via J̃ Σ J̃ᵀ where
//   J̃ = J_factor · U   (projection onto the non-zero eigenvectors of Ak).
// ─────────────────────────────────────────────────────────────────────────────
bool Marginalization::sparsifyVIO() {
    if (_n == 0) return false;

    const Eigen::Affine3d T_f_w  = _frame_to_keep->getWorld2FrameTransform();
    const Eigen::Matrix3d R_f_w  = T_f_w.rotation();
    const Eigen::Matrix3d t_skew = geometry::skewMatrix(T_f_w.translation());

    // ── Per-landmark absolute factors ─────────────────────────────────────────
    for (const auto& [type, lmks] : _lmk_to_keep) {
        for (const auto& lmk : lmks) {
            const int li = _map_lmk_idx.at(lmk);

            // Jacobian of   t_f_lmk = R_f_w · p_world + t_f_w
            // w.r.t. [δlmk (3), δpose_frame (6)]
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, _n);
            J.block(0, li,                                       3, 3) =  R_f_w;
            J.block(0, _map_frame_idx.at(_frame_to_keep),        3, 3) = -R_f_w * t_skew;
            J.block(0, _map_frame_idx.at(_frame_to_keep) + 3,    3, 3) =  R_f_w;

            const Eigen::MatrixXd J_tilde = J * _U;
            const Eigen::Matrix3d cov     = J_tilde * _Sigma.asDiagonal() * J_tilde.transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov.inverse());
            const Eigen::Vector3d S      = saes.eigenvalues().cwiseMax(0.0);
            const Eigen::Matrix3d inf_sqrt =
                saes.eigenvectors() * S.cwiseSqrt().asDiagonal() * saes.eigenvectors().transpose();

            _map_lmk_inf.emplace(lmk, inf_sqrt);
            _map_lmk_prior.emplace(lmk, T_f_w * lmk->getPose().translation());
        }
    }

    // ── Absolute frame factor ─────────────────────────────────────────────────
    const int fi = _map_frame_idx.at(_frame_to_keep);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(15, _n);
    J.block(0,  fi,      15, 15) = Eigen::MatrixXd::Identity(15, 15);
    J.block(0,  fi,      3,  3)  = T_f_w.rotation();
    J.block(0,  fi + 3,  3,  3)  = T_f_w.rotation();
    J.block(3,  fi + 3,  3,  3)  = T_f_w.rotation();

    const Eigen::MatrixXd J_tilde = J * _U;
    const Eigen::MatrixXd cov     = J_tilde * _Sigma.asDiagonal() * J_tilde.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(cov.inverse());
    const Eigen::VectorXd S      = saes.eigenvalues().cwiseMax(0.0);
    const Eigen::MatrixXd inf_sqrt =
        saes.eigenvectors() * S.cwiseSqrt().asDiagonal() * saes.eigenvectors().transpose();
    _map_frame_inf.emplace(_frame_to_keep, inf_sqrt);

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::sparsifyVO
//
// Replaces the dense landmark prior with a sparse Chow-Liu chain:
//   1. Order landmarks by maximum mutual information (greedy chain on off-diag Ak).
//   2. Select the landmark with minimum entropy as the chain anchor.
//   3. Add one absolute prior on the anchor.
//   4. Add relative priors between consecutive pairs in the chain.
// ─────────────────────────────────────────────────────────────────────────────
bool Marginalization::sparsifyVO() {
    if (_n == 0) return false;

    const auto& lmks = _lmk_to_keep.at("pointxd");
    const int N = static_cast<int>(lmks.size());
    if (N == 0) return false;

    // ── 1. Build off-diagonal information matrix (proxy for mutual information) ─
    Eigen::MatrixXd mi = Eigen::MatrixXd::Zero(N, N);
    for (int k = 0; k < N; k++)
        for (int l = k + 1; l < N; l++)
            mi(k, l) = mi(l, k) = computeOffDiag(lmks[k], lmks[l]);

    // ── 2. Greedy chain: start from the pair with maximum MI ──────────────────
    std::vector<std::shared_ptr<ALandmark>> chain;
    chain.reserve(N);

    int row, col;
    mi.maxCoeff(&row, &col);
    chain.push_back(lmks[row]);
    chain.push_back(lmks[col]);
    mi.col(row).setZero(); mi.row(row).setZero();
    mi.col(col).setZero();

    int cur = col;
    while (true) {
        int next;
        if (mi.row(cur).maxCoeff(&row, &next) == 0.0) break;
        chain.push_back(lmks[next]);
        mi.row(cur).setZero();
        mi.col(next).setZero();
        cur = next;
    }

    // ── 3. Anchor = landmark with minimum entropy in the chain ────────────────
    int anchor_idx = 0;
    double min_entropy = std::numeric_limits<double>::max();
    for (int k = 0; k < static_cast<int>(chain.size()); k++) {
        const double h = computeEntropy(chain[k]);
        if (h < min_entropy) { min_entropy = h; anchor_idx = k; }
    }
    _lmk_with_prior = chain[anchor_idx];

    // ── 4a. Absolute prior on the anchor ─────────────────────────────────────
    {
        const int li = _map_lmk_idx.at(_lmk_with_prior);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, _n);
        J.block(0, li, 3, 3) = Eigen::Matrix3d::Identity();

        const Eigen::MatrixXd J_tilde = J * _U;
        const Eigen::Matrix3d cov     = J_tilde * _Sigma.asDiagonal() * J_tilde.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov.inverse());
        const Eigen::Vector3d S = saes.eigenvalues().cwiseMax(0.0);
        _info_lmk  = saes.eigenvectors() * S.cwiseSqrt().asDiagonal() * saes.eigenvectors().transpose();
        _prior_lmk = _lmk_with_prior->getPose().translation();
    }

    // ── 4b. Relative priors along the chain ──────────────────────────────────
    _lmk_to_keep["pointxd"] = chain;  // update to chain order

    for (int k = 0; k + 1 < static_cast<int>(chain.size()); k++) {
        const auto& lk   = chain[k];
        const auto& lkp1 = chain[k + 1];
        const int ik     = _map_lmk_idx.at(lk);
        const int ikp1   = _map_lmk_idx.at(lkp1);

        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, _n);
        J.block(0, ik,   3, 3) =  Eigen::Matrix3d::Identity();
        J.block(0, ikp1, 3, 3) = -Eigen::Matrix3d::Identity();

        const Eigen::MatrixXd J_tilde = J * _U;
        const Eigen::Matrix3d cov     = J_tilde * _Sigma.asDiagonal() * J_tilde.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov.inverse());
        const Eigen::Vector3d S = saes.eigenvalues().cwiseMax(0.0);
        const Eigen::Matrix3d inf_sqrt =
            saes.eigenvectors() * S.cwiseSqrt().asDiagonal() * saes.eigenvectors().transpose();

        _map_lmk_inf.emplace(lkp1, inf_sqrt);
        _map_lmk_prior.emplace(lkp1, lk->getPose().translation() -
                                      lkp1->getPose().translation());
    }

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization::preMarginalizeRelative
//
// Alternative mode: marginalise all landmarks shared between frame0 and frame1
// to obtain a relative-pose factor between the two frames (pose-graph mode).
//
// Variable layout:
//   [shared_lmks (m DOF)] [frame0 pose | frame1 pose (n DOF)]
// ─────────────────────────────────────────────────────────────────────────────
void Marginalization::preMarginalizeRelative(std::shared_ptr<Frame>& frame0,
                                              std::shared_ptr<Frame>& frame1) {
    _frame_to_marg = nullptr;
    _frame_to_keep = nullptr;
    _lmk_to_keep.clear();
    _lmk_to_marg.clear();
    _map_frame_idx.clear();
    _map_lmk_idx.clear();
    _map_lmk_inf.clear();
    _map_lmk_prior.clear();
    _m = 0;
    _n = 0;

    int next_idx = 0;

    // Marginalise all landmarks visible from both frame0 and frame1.
    for (const auto& [type, lmks] : frame0->getLandmarks()) {
        const int lmk_dof = (type == "pointxd") ? 3 : 6;

        for (const auto& lmk : lmks) {
            if (lmk->isOutlier() || !lmk->isInMap() || !lmk->isInitialized())
                continue;

            bool seen_by_frame1 = false;
            for (const auto& weak_f : lmk->getFeatures()) {
                auto feat = weak_f.lock();
                if (feat && feat->getSensor()->getFrame() == frame1) {
                    seen_by_frame1 = true;
                    break;
                }
            }
            if (!seen_by_frame1) continue;

            _lmk_to_marg[type].push_back(lmk);
            _map_lmk_idx.emplace(lmk, next_idx);
            _m       += lmk_dof;
            next_idx += lmk_dof;
        }
    }

    // Keep both frame poses.
    _map_frame_idx.emplace(frame0, next_idx);
    _n       += frame0->getIMU() ? 15 : 6;
    next_idx += frame0->getIMU() ? 15 : 6;

    _map_frame_idx.emplace(frame1, next_idx);
    _n       += frame1->getIMU() ? 15 : 6;
    next_idx += frame1->getIMU() ? 15 : 6;
}

} // namespace orion_slam
