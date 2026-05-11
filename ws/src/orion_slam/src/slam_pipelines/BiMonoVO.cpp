#include "slam_pipelines/BiMonoVO.hpp"

#include "predictors/PnPPoseEstimator.hpp"
#include "tools/timer.hpp"
#include "tools/profiling.hpp"

namespace orion_slam {

static size_t countMatches(const typed_vec_match& m) {
    size_t n = 0;
    for (const auto& [type, v] : m) n += v.size();
    return n;
}

static size_t countLandmarks(const typed_vec_landmarks& lmks) {
    size_t n = 0;
    for (const auto& [type, v] : lmks) n += v.size();
    return n;
}


// ─────────────────────────────────────────────────────────────────────────────
bool BiMonoVO::init() {
    std::cout << "BiMonoVO initialization..." << std::endl;

    std::shared_ptr<Frame> frame;
    while (!frame)
        frame = _slam_param->getDataProvider()->next();

    frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    frame->setPrior(Eigen::Affine3d::Identity(), Vector6d::Ones());
    frame->setVelocity(Vector6d::Zero());

    _pipeline_core->detectFeatures(frame->getSensors().at(0));

    typed_vec_match matches_in_frame, matches_in_frame_with_lmk;
    _pipeline_core->trackFeatures(frame->getSensors().at(0),
                                  frame->getSensors().at(1),
                                  matches_in_frame,
                                  matches_in_frame_with_lmk,
                                  frame->getSensors().at(0)->getFeatures());

    _pipeline_core->initLandmarksInFrame(frame, matches_in_frame);
    _slam_param->getOptimizerFrontEnd()->landmarkOptimization(frame);

    frame->setKeyFrame();
    _local_map->addFrame(frame);

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
bool BiMonoVO::frontEndStep() {
    std::cout << "BiMonoVO front-end step..." << std::endl;

    const bool profiling = _slam_param->getConfig().enable_profiling;
    auto& stats = _pipeline_core->stats_;
    double _dummy = 0;

    // ── Get frames ────────────────────────────────────────────────────────────
    std::shared_ptr<Frame> frame, prev_frame;
    prev_frame = _local_map->getLastFrame();
    while (!frame)
        frame = _slam_param->getDataProvider()->next();

    frame->setWorld2FrameTransform(prev_frame->getWorld2FrameTransform());
    frame->setVelocity(prev_frame->getVelocity());
    if (profiling) stats.timestamp_ns = frame->getTimestamp();

    // ── Temporal tracking ─────────────────────────────────────────────────────
    typed_vec_match matches_in_time, matches_in_time_with_lmk;
    {
        timer::ScopedTimerTo t(profiling ? stats.match_time_t : _dummy);
        _pipeline_core->trackFeatures(prev_frame->getSensors().at(0),
                                      frame->getSensors().at(0),
                                      matches_in_time,
                                      matches_in_time_with_lmk,
                                      prev_frame->getSensors().at(0)->getFeatures());
    }
    if (profiling) stats.matches_time = static_cast<double>(countMatches(matches_in_time_with_lmk));

    // ── Pose prediction ───────────────────────────────────────────────────────
    bool predict_success;
    {
        timer::ScopedTimerTo t(profiling ? stats.predict_t : _dummy);
        predict_success = _pipeline_core->predictPose(prev_frame, frame, matches_in_time_with_lmk);
    }
    std::cout << "Pose prediction " << (predict_success ? "succeeded" : "failed (const vel)") << std::endl;

    if (_pipeline_core->isTrackingLost()) {
        std::cout << "--- TRACKING LOST: reinitializing ---" << std::endl;
        onTrackingLost(frame);
        return true;
    }

    // ── Outlier removal ───────────────────────────────────────────────────────
    {
        timer::ScopedTimerTo t(profiling ? stats.filter_t : _dummy);
        auto cam_prev = prev_frame->getSensors().at(0);
        auto cam_curr = frame->getSensors().at(0);
        matches_in_time = _pipeline_core->epipolarFiltering(cam_prev, cam_curr, matches_in_time);
        _pipeline_core->outlierRemoval(frame, matches_in_time, matches_in_time_with_lmk);
    }
    {
        timer::ScopedTimerTo t(profiling ? stats.clean_t : _dummy);
        _pipeline_core->cleanFeatures(frame);
    }

    // ── Keyframe selection ────────────────────────────────────────────────────
    typed_vec_match matches_in_frame;
    auto lastKF = _local_map->getLastFrame();
    if (_pipeline_core->shouldInsertKeyframe(frame, lastKF, matches_in_time_with_lmk)) {

        if (profiling) { stats.is_keyframe = true; stats.nkeyframes++; }

        {
            timer::ScopedTimerTo t(profiling ? stats.detect_t : _dummy);
            _pipeline_core->detectFeatures(frame->getSensors().at(0));
            _pipeline_core->recoverFeatureFromMapLandmarks(_local_map, frame);
        }

        {
            timer::ScopedTimerTo t(profiling ? stats.match_frame_t : _dummy);
            typed_vec_match matches_in_frame_with_lmk;
            _pipeline_core->trackFeatures(frame->getSensors().at(0),
                                          frame->getSensors().at(1),
                                          matches_in_frame,
                                          matches_in_frame_with_lmk,
                                          frame->getSensors().at(0)->getFeatures());
            matches_in_frame = _pipeline_core->epipolarFiltering(frame->getSensors().at(0),
                                                                 frame->getSensors().at(1),
                                                                 matches_in_frame);
        }
        if (profiling) stats.matches_frame = static_cast<double>(countMatches(matches_in_frame));

        {
            timer::ScopedTimerTo t(profiling ? stats.lmk_init_t : _dummy);
            _pipeline_core->initLandmarksInTime(frame, matches_in_time);
            _pipeline_core->initLandmarksInFrame(frame, matches_in_frame);
        }

        {
            timer::ScopedTimerTo t(profiling ? stats.frame_opt_t : _dummy);
            _slam_param->getOptimizerFrontEnd()->landmarkOptimization(frame);
            _slam_param->getOptimizerFrontEnd()->singleFrameOptimization(frame);
        }

        frame->setKeyFrame();
        _local_map->addFrame(frame);
        _new_kf_inserted = true;
    }

    if (profiling) {
        stats.lmk_inmap = static_cast<double>(countLandmarks(_local_map->getLandmarks()));
        const double ts_s = static_cast<double>(frame->getTimestamp()) * 1e-9;
        profiling::logPoseTUM(_TRAJ_TUM_PATH, ts_s, frame->getFrame2WorldTransform());
        stats.saveCSV(_STATS_CSV_PATH);
    }

    _pipeline_core->display_.frame_to_display                = frame;
    _pipeline_core->display_.matches_in_time_to_display      = matches_in_time;
    _pipeline_core->display_.matches_in_frame_to_display     = matches_in_frame;
    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
bool BiMonoVO::backEndStep() {
    _is_init = true;

    if (!_new_kf_inserted || _local_map->getMapSize() < 2)
        return true;

    _new_kf_inserted = false;

    const bool profiling = _slam_param->getConfig().enable_profiling;
    auto& stats   = _pipeline_core->stats_;
    double _dummy = 0;
    auto   backend = _slam_param->getOptimizerBackEnd();

    {
        timer::ScopedTimerTo t(profiling ? stats.wdw_opt_t : _dummy);
        backend->slidingWindowOptimization(_local_map, _local_map->getFixedFrameNumber());
    }

    if (profiling) {
        stats.opt_cost_initial = backend->_last_cost_initial;
        stats.opt_cost_final   = backend->_last_cost_final;
        stats.opt_iterations   = backend->_last_iterations;
        stats.prior_rank       = backend->_last_prior_rank;
        stats.prior_n          = backend->_last_prior_n;
        stats.saveCSV(_STATS_CSV_PATH);
    }

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
void BiMonoVO::onTrackingLost(std::shared_ptr<Frame>& frame) {
    _local_map->reset();
    _pipeline_core->resetTrackingState();

    _pipeline_core->cleanFeatures(frame);
    _pipeline_core->detectFeatures(frame->getSensors().at(0));

    typed_vec_match matches_in_frame, matches_in_frame_with_lmk;
    _pipeline_core->trackFeatures(frame->getSensors().at(0),
                                  frame->getSensors().at(1),
                                  matches_in_frame,
                                  matches_in_frame_with_lmk,
                                  frame->getSensors().at(0)->getFeatures());

    _pipeline_core->initLandmarksInFrame(frame, matches_in_frame);
    _slam_param->getOptimizerFrontEnd()->landmarkOptimization(frame);

    frame->setKeyFrame();
    _local_map->addFrame(frame);
}


} // namespace orion_slam
