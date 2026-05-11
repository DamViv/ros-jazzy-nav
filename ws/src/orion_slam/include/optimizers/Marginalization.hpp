#pragma once

#include <ceres/ceres.h>
#include <unordered_map>

#include "data/Frame.hpp"
#include "data/landmarks/ALandmark.hpp"
#include "data/maps/LocalMap.hpp"
#include "tools/typedefs.hpp"

namespace orion_slam {

// ─────────────────────────────────────────────────────────────────────────────
// MarginalizationBlockInfo
//
// Wraps a Ceres CostFunction together with:
//   - the global column indices of its parameter blocks in the (m+n)×(m+n)
//     information matrix (_parameter_idx)
//   - raw pointers to the current parameter values (_parameter_blocks)
//
// Calling Evaluate() linearises the factor at the current operating point and
// stores residuals + Jacobians ready for the Schur-complement computation.
// ─────────────────────────────────────────────────────────────────────────────
struct MarginalizationBlockInfo {

    MarginalizationBlockInfo(ceres::CostFunction*  cost_function,
                             std::vector<int>      parameter_idx,
                             std::vector<double*>  parameter_blocks)
        : _cost_function(cost_function),
          _parameter_idx(std::move(parameter_idx)),
          _parameter_blocks(std::move(parameter_blocks)) {}

    // Evaluate the wrapped factor at the current state.
    // Fills _residuals and _jacobians; allocates _raw_jacobians (freed by
    // Marginalization::computeSchurComplement after use).
    void Evaluate();

    ceres::CostFunction* _cost_function;
    std::vector<int>     _parameter_idx;      // column offsets in the full (m+n) matrix
    std::vector<double*> _parameter_blocks;   // pointers to current parameter values

    double** _raw_jacobians = nullptr;         // temporary, allocated in Evaluate()
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _jacobians;
    Eigen::VectorXd _residuals;
};


// ─────────────────────────────────────────────────────────────────────────────
// Marginalization
//
// Maintains the dense prior that summarises information from frames that have
// been dropped from the sliding window.
//
// Call sequence for each evicted frame F0 (next frame F1):
//   1. preMarginalize(F0, F1, prev_prior)
//        Classify variables: which to eliminate (m DOF) and which to keep (n DOF).
//   2. addMarginalizationBlock(...)   [once per factor touching F0]
//        Register every reprojection/IMU factor involving F0.
//   3. computeSchurComplement()
//        Build the (m+n)×(m+n) information matrix, eliminate the m DOF via
//        Schur complement, apply rank-revealing decomposition on the result.
//   4. computeJacobiansAndResiduals()
//        Convert the information-form prior (Ak, bk) to Ceres form:
//          J  = Λ^{½} Uᵀ
//          r₀ = −Λ^{−½} Uᵀ bk
//   5. (optional) sparsifyVO() / sparsifyVIO()
//        Replace the dense prior with a sparse chain of factors.
//   Then insert a MarginalizationFactor into the next optimisation problem.
// ─────────────────────────────────────────────────────────────────────────────
class Marginalization {
  public:

    // ── Step 1 : classify variables ───────────────────────────────────────────
    //
    // Scans the landmarks of frame0 and assigns each to one of three groups:
    //   • marginalize  — lonely landmarks (only seen from frame0)
    //   • keep         — landmarks also seen from other frames (will carry prior)
    //   • ignore       — outliers / uninitialised / no full stereo constraint
    //
    // frame1 is always added to the kept variables so that subsequent
    // marginalisations can propagate the prior onto frame poses (VO and VIO).
    //
    // marginalization_last carries the landmarks that survived the previous
    // marginalisation step and must be included here if still alive.
    void preMarginalize(std::shared_ptr<Frame>&          frame0,
                        std::shared_ptr<Frame>&          frame1,
                        std::shared_ptr<Marginalization>& marginalization_last);

    // Alternative mode: marginalise all landmarks shared between frame0 and
    // frame1 to produce a relative-pose factor between the two frames.
    // Used for pose-graph reduction, not for sliding-window BA.
    void preMarginalizeRelative(std::shared_ptr<Frame>& frame0,
                                std::shared_ptr<Frame>& frame1);

    void addMarginalizationBlock(std::shared_ptr<MarginalizationBlockInfo> block) {
        _marginalization_blocks.push_back(std::move(block));
    }

    // ── Steps 3 & 4 : compute the prior ──────────────────────────────────────

    // Build the full information matrix, apply Schur complement to eliminate
    // the m marginalised DOF, decompose the result.
    bool computeSchurComplement();

    // Convert the information-form prior (Ak, bk) into the Jacobian / residual
    // form used by Ceres inside MarginalizationFactor.
    bool computeJacobiansAndResiduals();

    // ── Step 5 (optional) : sparsification ───────────────────────────────────

    // VO variant: replace the dense landmark prior with a sparse Chow-Liu chain
    // of relative landmark factors.
    bool sparsifyVO();

    // VIO variant: replace the dense prior with per-landmark absolute factors
    // and an absolute frame factor.
    bool sparsifyVIO();

    // ── Information metrics (used internally by sparsifyVO) ───────────────────

    double computeEntropy         (std::shared_ptr<ALandmark> lmk);
    double computeMutualInformation(std::shared_ptr<ALandmark> lmk_i,
                                    std::shared_ptr<ALandmark> lmk_j);
    double computeOffDiag         (std::shared_ptr<ALandmark> lmk_i,
                                    std::shared_ptr<ALandmark> lmk_j);
    double computeKLD             (Eigen::MatrixXd A_p, Eigen::MatrixXd A_q);

    // ── Dimensions ───────────────────────────────────────────────────────────
    int _m      = 0;   // DOF to marginalise (frame0 + lonely landmarks)
    int _n      = 0;   // DOF to keep        (frame1 + shared landmarks)
    int _n_full = 0;   // effective rank of Ak after rank-revealing decomposition
    const double _eps = 1e-12;

    // ── Variable bookkeeping ─────────────────────────────────────────────────
    std::shared_ptr<Frame>   _frame_to_marg;  // frame being eliminated
    std::shared_ptr<Frame>   _frame_to_keep;  // frame receiving the prior
    typed_vec_landmarks      _lmk_to_keep;    // landmarks whose prior is kept
    typed_vec_landmarks      _lmk_to_marg;    // landmarks being eliminated

    // Column index of each variable in the full (m+n) state vector.
    // After computeSchurComplement(), indices are shifted to [0, n).
    std::unordered_map<std::shared_ptr<Frame>,     int>             _map_frame_idx;
    std::unordered_map<std::shared_ptr<ALandmark>, int>             _map_lmk_idx;
    std::unordered_map<std::shared_ptr<Frame>,     Eigen::MatrixXd> _map_frame_inf;

    std::vector<std::shared_ptr<MarginalizationBlockInfo>> _marginalization_blocks;

    // ── Sparsification results (filled by sparsifyVO / sparsifyVIO) ──────────
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Matrix3d> _map_lmk_inf;
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Vector3d> _map_lmk_prior;
    std::shared_ptr<ALandmark> _lmk_with_prior;
    Eigen::Matrix3d _info_lmk;
    Eigen::Vector3d _prior_lmk;

    // ── Dense prior matrices (filled by computeSchurComplement) ─────────────
    Eigen::MatrixXd _Ak;      // Schur-complement information matrix  (_n × _n)
    Eigen::VectorXd _bk;      // Schur-complement gradient            (_n × 1)
    Eigen::MatrixXd _Sigma_k; // covariance  = Ak^{-1}  (used by sparsification)
    Eigen::MatrixXd _U;       // eigenvectors of Ak (only columns with λ > eps)
    Eigen::VectorXd _Lambda;  // corresponding eigenvalues
    Eigen::VectorXd _Sigma;   // 1 / _Lambda

    // ── Ceres-form prior (filled by computeJacobiansAndResiduals) ────────────
    Eigen::MatrixXd _marginalization_jacobian;  // J  = Λ^{½} Uᵀ    (_n_full × _n)
    Eigen::VectorXd _marginalization_residual;  // r₀ = −Λ^{−½} Uᵀ bk (_n_full)

  private:
    // Accumulate  A += Σ Jᵀ J  and  b += Σ Jᵀ r  from a set of linearised
    // factors, spread over 4 threads.
    void computeInformationAndGradient(
        std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
        Eigen::MatrixXd& A, Eigen::VectorXd& b);

    // Thin eigendecomposition of A: discards eigenvectors with eigenvalue ≤ eps.
    void rankReveallingDecomposition(Eigen::MatrixXd  A,
                                     Eigen::MatrixXd& U,
                                     Eigen::VectorXd& d);
};


// ─────────────────────────────────────────────────────────────────────────────
// MarginalizationFactor
//
// Ceres CostFunction encoding the linearised prior around the stored operating
// point:
//
//   r(x) = r₀ + J · δx
//
// where δx is the perturbation of the kept variables from the linearisation
// point and J, r₀ are stored in the Marginalization object.
//
// Parameter blocks (in registration order):
//   [frame_to_keep — pose 6]
//   [frame_to_keep — velocity 3, bias_acc 3, bias_gyro 3]  ← VIO only
//   [lmk₀ — 3 or 6] [lmk₁ — 3 or 6] ...
// ─────────────────────────────────────────────────────────────────────────────
class MarginalizationFactor : public ceres::CostFunction {
  public:
    explicit MarginalizationFactor(std::shared_ptr<Marginalization> info)
        : _info(std::move(info)) {

        // ── Register frame parameter blocks ───────────────────────────────────
        // Every kept frame contributes a 6-DOF pose block.
        // VIO frames additionally have velocity (3) + bias_acc (3) + bias_gyro (3).
        if (_info->_frame_to_keep) {
            mutable_parameter_block_sizes()->push_back(6);           // pose (VO + VIO)
            if (_info->_frame_to_keep->getIMU()) {
                mutable_parameter_block_sizes()->push_back(3);       // velocity
                mutable_parameter_block_sizes()->push_back(3);       // bias_acc
                mutable_parameter_block_sizes()->push_back(3);       // bias_gyro
            }
        }

        // ── Register landmark parameter blocks ────────────────────────────────
        for (const auto& [type, lmks] : _info->_lmk_to_keep) {
            const int dof = (type == "pointxd") ? 3 : 6;
            for (size_t i = 0; i < lmks.size(); ++i)
                mutable_parameter_block_sizes()->push_back(dof);
        }

        set_num_residuals(_info->_n_full);
    }

    bool Evaluate(double const* const* parameters,
                  double*              residuals,
                  double**             jacobians) const override {

        const int n          = _info->_n_full;
        const auto& J_prior  = _info->_marginalization_jacobian;  // (n_full × n)
        const auto& r0       = _info->_marginalization_residual;  // (n_full)

        // ── Build δx : perturbation of all kept variables from their linearisation
        //              point, assembled from the Ceres parameter blocks.
        Eigen::VectorXd dx(_info->_n);
        dx.setZero();
        int blk = 0;

        // Frame perturbation
        if (_info->_frame_to_keep) {
            const int fi = _info->_map_frame_idx.at(_info->_frame_to_keep);

            dx.segment<6>(fi) =
                Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[blk++]);

            if (_info->_frame_to_keep->getIMU()) {
                dx.segment<3>(fi + 6)  = Eigen::Map<const Eigen::Vector3d>(parameters[blk++]);
                dx.segment<3>(fi + 9)  = Eigen::Map<const Eigen::Vector3d>(parameters[blk++]);
                dx.segment<3>(fi + 12) = Eigen::Map<const Eigen::Vector3d>(parameters[blk++]);
            }
        }

        // Landmark perturbations
        for (const auto& [type, lmks] : _info->_lmk_to_keep) {
            const int dof = (type == "pointxd") ? 3 : 6;
            for (const auto& lmk : lmks) {
                const int li = _info->_map_lmk_idx.at(lmk);
                if (li != -1)
                    dx.segment(li, dof) =
                        Eigen::Map<const Eigen::VectorXd>(parameters[blk], dof);
                blk++;  // always advance — block registered even for li == -1
            }
        }

        // ── Residual : r = r₀ + J · δx ────────────────────────────────────────
        Eigen::Map<Eigen::VectorXd>(residuals, n) = r0 + J_prior * dx;

        if (!jacobians)
            return true;

        // ── Jacobian blocks ────────────────────────────────────────────────────
        blk = 0;

        // Helper: write one Jacobian block from a column slice of J_prior.
        auto fillJac = [&](int blk_id, int col_start, int cols) {
            if (!jacobians[blk_id]) return;
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                jac(jacobians[blk_id], n, cols);
            jac = J_prior.middleCols(col_start, cols);
        };

        // Frame Jacobians
        if (_info->_frame_to_keep) {
            const int fi = _info->_map_frame_idx.at(_info->_frame_to_keep);
            fillJac(blk++, fi,      6);   // pose
            if (_info->_frame_to_keep->getIMU()) {
                fillJac(blk++, fi + 6,  3);  // velocity
                fillJac(blk++, fi + 9,  3);  // bias_acc
                fillJac(blk++, fi + 12, 3);  // bias_gyro
            }
        }

        // Landmark Jacobians
        for (const auto& [type, lmks] : _info->_lmk_to_keep) {
            const int dof = (type == "pointxd") ? 3 : 6;
            for (const auto& lmk : lmks) {
                const int li = _info->_map_lmk_idx.at(lmk);
                if (li != -1)
                    fillJac(blk, li, dof);
                blk++;  // always advance
            }
        }

        return true;
    }

  private:
    std::shared_ptr<Marginalization> _info;
};

} // namespace orion_slam
