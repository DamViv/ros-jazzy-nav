#include "predictors/ESKFPoseEstimator.hpp"
#include "predictors/PnPPoseEstimator.hpp"


namespace orion_slam {


// Function to compute the Jacobian of the switch from homogeneous point to 2D point
Eigen::MatrixXd ESKFPoseEstimator::jac_homogeneous(const Eigen::Vector3d &point) {
    Eigen::MatrixXd jac(2, 3);
    double inv_z    = 1.0 / point(2);
    double inv_z_sq = inv_z * inv_z;

    jac(0, 0) = inv_z;
    jac(0, 1) = 0;
    jac(0, 2) = -point(0) * inv_z_sq;

    jac(1, 0) = 0;
    jac(1, 1) = inv_z;
    jac(1, 2) = -point(1) * inv_z_sq;

    return jac;
}

// Function to compute the Jacobian of the update of the pose T \delta \tau w.r.t the delta
Eigen::MatrixXd ESKFPoseEstimator::jac_delta_update(const Eigen::Vector3d &dtheta, const Eigen::Matrix3d &rotation) {
    Eigen::MatrixXd jr = geometry::so3_rightJacobian(dtheta);
    Eigen::MatrixXd jac(6, 6);

    jac.block<3, 3>(0, 0) = jr;
    jac.block<3, 3>(0, 3) = Eigen::MatrixXd::Zero(3, 3);
    jac.block<3, 3>(3, 0) = Eigen::MatrixXd::Zero(3, 3);
    jac.block<3, 3>(3, 3) = rotation;

    return jac;
}

std::tuple<Eigen::Vector2d, Eigen::MatrixXd, Eigen::MatrixXd> ESKFPoseEstimator::jac_projection(const Eigen::Vector3d& point_3d,
                                                                                                const Eigen::Matrix3d& rotation,
                                                                                                const Eigen::Vector3d& translation,
                                                                                                const Eigen::Vector3d& dtheta) {

    // Point in camera frame
    Eigen::Vector3d p_cam = rotation * point_3d + translation;

    // 2D normalized projection
    Eigen::Vector2d proj_2d(p_cam(0) / p_cam(2), p_cam(1) / p_cam(2));

    // J_hom : 2x3, jacobienne de la division perspective
    Eigen::MatrixXd J_hom = jac_homogeneous(p_cam);

    // J_pose : 3x6, jacobienne de (R*p + t) par rapport à [δθ_body, δt_body]
    // Right SE3 perturbation: T_new = T * exp([δω; δv]) → t_new = t + R*δv
    // So ∂(R*p + t)/∂δv = R  (not I)
    Eigen::MatrixXd J_pose(3, 6);
    J_pose.block<3,3>(0,0) = -rotation * geometry::skewMatrix(point_3d); // jacobien wrt δθ
    J_pose.block<3,3>(0,3) = rotation;                                    // jacobien wrt δt_body

    // Jacobienne finale par rapport à l'état [δθ, δt]
    Eigen::MatrixXd J_T = J_hom * J_pose; 

    // Jacobienne par rapport au point 3D
    Eigen::MatrixXd J_p = J_hom * rotation;

    return {proj_2d, J_T, J_p};
}



bool ESKFPoseEstimator::predictFromIMU(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2,
    Eigen::Affine3d& T_cam2_cam1,
    Eigen::Matrix<double, 6, 6> &P) {

    auto imu = frame2->getIMU();
    if (!imu)
        return false;

    double dt = (frame2->getTimestamp() - frame1->getTimestamp()) * 1e-9;
    Eigen::Matrix3d R1 = frame1->getFrame2WorldTransform().rotation();

    // --- Correction rotation ---
    Eigen::Vector3d err_r = geometry::log_so3(
        T_cam2_cam1.linear().transpose() * imu->getDeltaTransform().linear());

    Eigen::Matrix3d J_delta = geometry::so3_rightJacobian(err_r).inverse();
    Eigen::Matrix3d J_rot   = -geometry::so3_leftJacobian(err_r).inverse();

    Eigen::Matrix3d S_r = J_delta * imu->getCovariance().block<3,3>(0,0) * J_delta.transpose() * _imu_weight
                        + J_rot   * P.block<3,3>(0,0)             * J_rot.transpose();

    Eigen::Matrix3d K_r = P.block<3,3>(0,0) * S_r.inverse();

    T_cam2_cam1.linear() = T_cam2_cam1.linear() * geometry::exp_so3(K_r * err_r);
    P.block<3,3>(0,0)    = (Eigen::Matrix3d::Identity() - K_r) * P.block<3,3>(0,0);

    // --- Correction translation ---
    Eigen::Vector3d delta_p_est =
        T_cam2_cam1.translation()
        - R1.transpose() * frame1->getLinearVelocity() * dt
        - 0.5 * R1.transpose() * _gravity * dt * dt;

    Eigen::Vector3d err_t = imu->getDeltaTransform().translation() - delta_p_est;

    Eigen::Matrix3d S_t = P.block<3,3>(3,3)
                        + imu->getCovariance().block<3,3>(6,6) * _imu_weight;

    Eigen::Matrix3d K_t = P.block<3,3>(3,3) * S_t.inverse();

    T_cam2_cam1.translation() = T_cam2_cam1.translation() + K_t * err_t;
    P.block<3,3>(3,3)         = (Eigen::Matrix3d::Identity() - K_t) * P.block<3,3>(3,3);

    // --- Correction vitesse (stockée dans le frame) ---
    Eigen::Vector3d v_cst =
        (frame2->getFrame2WorldTransform().translation()
       - frame1->getFrame2WorldTransform().translation()) / dt;

    Eigen::Vector3d err_v = imu->getDeltaVelocity()
        - R1.transpose() * (v_cst - frame1->getLinearVelocity() - _gravity * dt);

    Eigen::Matrix3d S_v = Eigen::Matrix3d::Identity()
                        + imu->getCovariance().block<3,3>(3,3) * _imu_weight;

    Eigen::Matrix3d K_v = S_v.inverse();    
    
    frame2->setLinearVelocity(v_cst + K_v * err_v);

    return true;
}



void ESKFPoseEstimator::extractObservations(
    const std::shared_ptr<Frame>& frame1,
    const vec_match& matches,
    std::vector<Eigen::Vector3d>& p3d_cam1,
    std::vector<Eigen::Vector2d>& p2d_cam2) {

    static constexpr double MIN_DEPTH = 0.1;

    Eigen::Affine3d T_cam1_world = frame1->getSensors().at(0)->getWorld2SensorTransform();

    p3d_cam1.clear();
    p2d_cam2.clear();
    p3d_cam1.reserve(matches.size());
    p2d_cam2.reserve(matches.size());

    for (const auto& [f1, f2] : matches) {
        auto lmk = f1->getLandmark().lock();
        if (!lmk || !lmk->isInitialized())
            continue;

        // Point 3D dans le repère cam1
        Eigen::Vector3d p3d = T_cam1_world * lmk->getPose().translation();
        if (p3d.z() < MIN_DEPTH)
            continue;

        // Observation normalisée dans cam2 — copy by value, getBearingVectors() may return a temporary
        Eigen::Vector3d ray = f2->getBearingVectors().at(0);
        if (ray.z() < MIN_DEPTH)
            continue;

        p3d_cam1.push_back(p3d);
        p2d_cam2.push_back(ray.head<2>() / ray.z());
    }
}



void ESKFPoseEstimator::updateFromVisual(
    const std::vector<Eigen::Vector3d>& p3d_cam1,
    const std::vector<Eigen::Vector2d>& p2d_cam2,
    Eigen::Affine3d& T_cam2_cam1,
    Eigen::Matrix<double, 6, 6>& P) {

    static constexpr double MIN_DEPTH        = 0.1;
    static constexpr double MAX_RESIDUAL_NRM = 0.5;

    const Eigen::Matrix2d R_noise = _noise_pixel * Eigen::Matrix2d::Identity();

    for (size_t i = 0; i < p3d_cam1.size(); ++i) {

        Eigen::Vector3d p_cam = T_cam2_cam1.linear() * p3d_cam1[i] + T_cam2_cam1.translation();
        if (p_cam.z() < MIN_DEPTH)
            continue;

        auto [proj, J_T, J_p] = jac_projection(p3d_cam1[i],
                                               T_cam2_cam1.linear(),
                                               T_cam2_cam1.translation(),
                                               Eigen::Vector3d::Zero());

        Eigen::Vector2d residual = p2d_cam2[i] - proj;
        if (residual.norm() > MAX_RESIDUAL_NRM)
            continue;

        Eigen::Matrix2d S     = J_T * P * J_T.transpose() + R_noise;
        Eigen::Matrix<double, 6, 2> K  = P * J_T.transpose() * S.inverse();
        Eigen::Matrix<double, 6, 1> dx = K * residual;

        if (!dx.allFinite())
            continue;

        Eigen::Matrix<double, 6, 6> IKH = Eigen::Matrix<double, 6, 6>::Identity() - K * J_T;
        P = IKH * P * IKH.transpose() + K * R_noise * K.transpose();

        Eigen::Affine3d dT = Eigen::Affine3d::Identity();
        dT.linear()        = geometry::exp_so3(dx.head<3>());
        dT.translation()   = dx.segment<3>(3);
        T_cam2_cam1        = T_cam2_cam1 * dT;
    }
}



bool ESKFPoseEstimator::estimateTransformBetween(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2,
    vec_match& matches,
    Eigen::Affine3d& dT,
    Eigen::MatrixXd &covdT) {

    // --------------------------------------------------
    // Step 1: PnP initialization — provides robust initial pose
    // --------------------------------------------------
    PnPPoseEstimator pnp;
    Eigen::MatrixXd cov_pnp = Eigen::MatrixXd::Identity(6, 6);
    if (!pnp.estimateTransformBetween(frame1, frame2, matches, dT, cov_pnp))
        return false;

    if (!dT.matrix().allFinite()) {
        std::cout << "ESKF: PnP returned non-finite dT, aborting" << std::endl;
        return false;
    }

    

    // Convert dT (frame space: ^{F_prev} T_{F_curr}) → camera space (^{C2} T_{C1})
    // T_cam_f = ^C T_F (sensor extrinsic: body-to-camera)
    Eigen::Affine3d T_cam_f = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d T_cam2_cam1 = T_cam_f * dT.inverse() * T_cam_f.inverse();

    std::cout << " pnp: \n " << T_cam2_cam1.matrix() << std::endl;

    // --------------------------------------------------
    // Step 2: Initial covariance after PnP (tight, trust PnP)
    // --------------------------------------------------
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
    P.block<3,3>(0,0) = _init_cov_rot   * Eigen::Matrix3d::Identity();
    P.block<3,3>(3,3) = _init_cov_trans * Eigen::Matrix3d::Identity();

    // --------------------------------------------------
    // Step 3: Optional IMU refinement
    // --------------------------------------------------
    // if (frame1->getIMU())
    //     predictFromIMU(frame1, frame2, T_cam2_cam1, P);

    // --------------------------------------------------
    // Step 4: Extract visual observations
    // --------------------------------------------------
    std::vector<Eigen::Vector3d> p3d_cam1;
    std::vector<Eigen::Vector2d> p2d_cam2;
    extractObservations(frame1, matches, p3d_cam1, p2d_cam2);

    if (p3d_cam1.size() < _min_matches) {
        std::cout << "ESKF: not enough visual matches (" << p3d_cam1.size() << ")" << std::endl;
        return false;
    }

    // --------------------------------------------------
    // Step 5: ESKF visual update — refines pose
    // --------------------------------------------------
    updateFromVisual(p3d_cam1, p2d_cam2, T_cam2_cam1, P);

    std::cout << " ekf: \n " << T_cam2_cam1.matrix() << std::endl;

    // Velocity from ESKF-refined translation (cam1 frame → world frame)
    double dt = (frame2->getTimestamp() - frame1->getTimestamp()) * 1e-9;
    if (dt > 0) {
        Eigen::Matrix3d R_world_cam1 = frame1->getFrame2WorldTransform().rotation();
        frame2->setLinearVelocity(R_world_cam1 * T_cam2_cam1.translation() / dt);
    }

    // Convert back to frame space: ^{F_prev} T_{F_curr}
    dT    = T_cam_f.inverse() * T_cam2_cam1.inverse() * T_cam_f;
    covdT = P;
    return true;
}






bool ESKFPoseEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                              typed_vec_match &typed_matches, Eigen::Affine3d &dT, Eigen::MatrixXd &covdT) {

  return estimateTransformBetween(frame1, frame2, typed_matches["pointxd"], dT, covdT);
}




} // namespace orion_slam

