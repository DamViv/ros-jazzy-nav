#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <Eigen/Geometry>

#include "data/Frame.hpp"
#include "tools/typedefs.hpp"

namespace orion_slam {

namespace profiling {


class Display {
    public:
        Display(){};

        std::shared_ptr<Frame> frame_to_display;
        typed_vec_match matches_in_time_to_display;
        typed_vec_match matches_in_frame_to_display;
};


// ─────────────────────────────────────────────────────────────────────────────
// Forward declaration — Stats is defined below; StatsAccumulator::update is
// defined inline after Stats is complete (avoids circular dependency).
// ─────────────────────────────────────────────────────────────────────────────
struct Stats;


// ─────────────────────────────────────────────────────────────────────────────
// StatsAccumulator
//
// Accumulates per-frame Stats over a full run.
// Lives INSIDE Stats so there is no separate object to manage.
// NOT reset by Stats::reset() — it survives the whole run.
//
// Call stats.commit()       — once per logical frame (front+back end done)
// Call stats.saveSummary()  — periodically or at end of run
// ─────────────────────────────────────────────────────────────────────────────
struct StatsAccumulator {

    struct Field {
        int    n    = 0;
        double sum  = 0.0;
        double sum2 = 0.0;
        double vmin =  std::numeric_limits<double>::max();
        double vmax = -std::numeric_limits<double>::max();

        void update(double v) {
            if (v == 0.0) return;
            n++;
            sum  += v;
            sum2 += v * v;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        double mean()   const { return n ? sum / n : 0.0; }
        double stddev() const {
            if (n < 2) return 0.0;
            double m = mean();
            return std::sqrt(std::max(0.0, sum2 / n - m * m));
        }
        double min() const { return n ? vmin : 0.0; }
        double max() const { return n ? vmax : 0.0; }
    };

    Field detect_t, match_time_t, match_frame_t, predict_t, filter_t;
    Field frame_opt_t, wdw_opt_t, marg_t, clean_t;
    Field matches_time, matches_frame, lmk_inmap, lmk_new;
    Field opt_cost_initial, opt_cost_final, opt_iterations;
    Field prior_rank, prior_n;

    int total_frames = 0;
    int total_kf     = 0;

    // Defined after Stats (below).
    void update(const Stats& s);

    void saveSummary(const std::string& path) const {
        std::ofstream f(path);
        if (!f) return;

        auto row = [&](const char* name, const Field& fld) {
            if (fld.n == 0) return;
            f << std::left  << std::setw(22) << name
              << std::right << std::fixed << std::setprecision(5)
              << std::setw(11) << fld.mean()
              << std::setw(10) << fld.stddev()
              << std::setw(10) << fld.min()
              << std::setw(10) << fld.max()
              << "   (" << fld.n << ")\n";
        };

        f << "═══════════════════════════════════════════════════════════════\n";
        f << "  OrionSLAM — Run Summary\n";
        f << "═══════════════════════════════════════════════════════════════\n";
        f << "  Frames: " << total_frames << "   KeyFrames: " << total_kf << "\n";
        f << "───────────────────────────────────────────────────────────────\n";
        f << std::left  << std::setw(22) << "Metric"
          << std::right
          << std::setw(11) << "mean"
          << std::setw(10) << "std"
          << std::setw(10) << "min"
          << std::setw(10) << "max"
          << "   (n)\n";
        f << "───────────────────────────────────────────────────────────────\n";

        f << "── Timing (s) ──────────────────────────────────────────────────\n";
        row("detect",          detect_t);
        row("track_time",      match_time_t);
        row("track_frame",     match_frame_t);
        row("predict",         predict_t);
        row("filter+clean",    filter_t);
        row("frame_optim",     frame_opt_t);
        row("window_optim",    wdw_opt_t);
        row("marginalization", marg_t);

        f << "── Match / Landmark counts ─────────────────────────────────────\n";
        row("matches_time",    matches_time);
        row("matches_frame",   matches_frame);
        row("lmk_in_map",      lmk_inmap);
        row("lmk_new/kf",      lmk_new);

        f << "── Optimization quality ────────────────────────────────────────\n";
        row("cost_initial",    opt_cost_initial);
        row("cost_final",      opt_cost_final);
        row("iterations",      opt_iterations);
        row("prior_rank",      prior_rank);
        row("prior_n",         prior_n);

        if (opt_cost_initial.n && opt_cost_initial.mean() > 0) {
            f << "── Cost reduction ──────────────────────────────────────────────\n";
            f << "  cost_final / cost_initial (mean): "
              << std::fixed << std::setprecision(4)
              << opt_cost_final.mean() / opt_cost_initial.mean() << "\n";
        }

        f << "═══════════════════════════════════════════════════════════════\n";
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Stats — per-frame performance metrics
//
// reset()        — clear per-frame fields (accumulator untouched)
// commit()       — push this frame into the accumulator (call once per frame)
// saveCSV()      — append a raw row to the per-frame CSV
// saveSummary()  — write aggregate mean/std/min/max summary
// ─────────────────────────────────────────────────────────────────────────────
struct Stats {

    // ── Timing (seconds) ─────────────────────────────────────────────────────
    double detect_t      = 0;
    double lmk_init_t    = 0;
    double lmk_resur_t   = 0;
    double match_frame_t = 0;
    double match_time_t  = 0;
    double predict_t     = 0;
    double filter_t      = 0;
    double processing_t  = 0;
    double frame_opt_t   = 0;
    double wdw_opt_t     = 0;
    double clean_t       = 0;
    double marg_t        = 0;

    // ── Counts ────────────────────────────────────────────────────────────────
    double matches_frame = 0;
    double matches_time  = 0;
    double removed_feat  = 0;
    double lmk_inmap     = 0;
    double lmk_new       = 0;
    double resur_lmk     = 0;

    // ── Optimization quality ──────────────────────────────────────────────────
    double opt_cost_initial = 0;
    double opt_cost_final   = 0;
    int    opt_iterations   = 0;
    int    prior_rank       = 0;
    int    prior_n          = 0;

    // ── Metadata ──────────────────────────────────────────────────────────────
    uint64_t timestamp_ns = 0;
    bool     is_keyframe  = false;
    uint     nframes      = 0;
    uint     nkeyframes   = 0;

    // ── Accumulator (persists across reset()) ─────────────────────────────────
    StatsAccumulator accum;

    // Reset per-frame fields only — accumulator is preserved.
    void reset() {
        detect_t = lmk_init_t = lmk_resur_t = match_frame_t = match_time_t = 0;
        predict_t = filter_t = processing_t = frame_opt_t = wdw_opt_t = 0;
        clean_t = marg_t = 0;
        matches_frame = matches_time = removed_feat = lmk_inmap = lmk_new = resur_lmk = 0;
        opt_cost_initial = opt_cost_final = 0;
        opt_iterations = prior_rank = prior_n = 0;
        timestamp_ns = 0;
        is_keyframe  = false;
    }

    // Push the current frame into the accumulator.
    // Call once per logical frame after all fields (front-end + back-end) are set.
    void commit() { accum.update(*this); }

    // Append one row to the per-frame CSV.
    void saveCSV(const std::string& path) const {
        const bool write_header = !std::ifstream(path).good();
        std::ofstream f(path, std::ios::app);
        if (!f) return;
        if (write_header)
            f << "frame,kf,timestamp_ns,"
              << "detect_t,match_time_t,match_frame_t,predict_t,filter_t,"
              << "frame_opt_t,wdw_opt_t,marg_t,clean_t,"
              << "matches_time,matches_frame,lmk_inmap,lmk_new,resur_lmk,removed_feat,"
              << "opt_cost_initial,opt_cost_final,opt_iterations,prior_rank,prior_n\n";
        f << std::fixed << std::setprecision(6)
          << nframes << "," << is_keyframe << "," << timestamp_ns << ","
          << detect_t      << "," << match_time_t  << "," << match_frame_t << ","
          << predict_t     << "," << filter_t       << ","
          << frame_opt_t   << "," << wdw_opt_t      << "," << marg_t << "," << clean_t << ","
          << matches_time  << "," << matches_frame  << "," << lmk_inmap << "," << lmk_new << ","
          << resur_lmk     << "," << removed_feat   << ","
          << opt_cost_initial << "," << opt_cost_final << "," << opt_iterations << ","
          << prior_rank    << "," << prior_n        << "\n";
    }

    // Write the aggregate summary to a text file.
    void saveSummary(const std::string& path) const { accum.saveSummary(path); }
};


// StatsAccumulator::update defined here so Stats is complete.
inline void StatsAccumulator::update(const Stats& s) {
    total_frames++;
    if (s.is_keyframe) total_kf++;

    detect_t.update(s.detect_t);
    match_time_t.update(s.match_time_t);
    match_frame_t.update(s.match_frame_t);
    predict_t.update(s.predict_t);
    filter_t.update(s.filter_t);
    frame_opt_t.update(s.frame_opt_t);
    wdw_opt_t.update(s.wdw_opt_t);
    marg_t.update(s.marg_t);
    clean_t.update(s.clean_t);

    matches_time.update(s.matches_time);
    matches_frame.update(s.matches_frame);
    lmk_inmap.update(s.lmk_inmap);
    lmk_new.update(s.lmk_new);

    opt_cost_initial.update(s.opt_cost_initial);
    opt_cost_final.update(s.opt_cost_final);
    opt_iterations.update(static_cast<double>(s.opt_iterations));
    prior_rank.update(static_cast<double>(s.prior_rank));
    prior_n.update(static_cast<double>(s.prior_n));
}


// ─────────────────────────────────────────────────────────────────────────────
// logPoseTUM
//
// Appends one line to a TUM-format trajectory file.
// Format: "timestamp_s  tx ty tz  qx qy qz qw"  (T_world_frame)
//
// Compatible with: evo_ape, evo_rpe (pip install evo)
//   evo_ape tum ground_truth.txt /tmp/orion_slam_traj.txt --plot
// ─────────────────────────────────────────────────────────────────────────────
inline void logPoseTUM(const std::string& path,
                       double timestamp_s,
                       const Eigen::Affine3d& T_world_frame) {
    std::ofstream f(path, std::ios::app);
    if (!f) return;
    const Eigen::Vector3d    t = T_world_frame.translation();
    const Eigen::Quaterniond q(T_world_frame.rotation());
    f << std::fixed << std::setprecision(9)
      << timestamp_s << " "
      << t.x() << " " << t.y() << " " << t.z() << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}


} // namespace profiling

} // namespace orion_slam
