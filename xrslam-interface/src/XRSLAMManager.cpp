#include "XRSLAMManager.h"

#include <xrslam/core/frontend_worker.h>
#include <xrslam/core/feature_tracker.h>
#include <xrslam/map/map.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/track.h>
#include <xrslam/geometry/stereo.h>
#include <xrslam/core/place_graph_4dof_shadow.h>
#include <xrslam/place_recognition.h>
#include <xrslam/extra/orb_local_descriptor_extractor.h>
#if defined(XRSLAM_HAS_PIDNET)
#include <xrslam/extra/pidnet_semantic_worker.h>
#endif
#include <xrslam/core/relocalization_runtime_profile.h>

#if defined(XRSLAM_HAS_EIGENPLACES)
#include <xrslam/extra/eigenplaces_descriptor_extractor.h>
#endif

#if defined(XRSLAM_HAS_USEARCH)
#include <xrslam/extra/usearch_place_database.h>
#endif

#include <algorithm>
#include <cstring>
#include <vector>
#include <sstream>
#include <optional>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
#include <stdexcept>

#define XRSLAM_VERSION "0.1.0"

namespace xrslam {
XRSLAMManager &XRSLAMManager::Instance() {
    static XRSLAMManager SLAMManagerInstance;
    return SLAMManagerInstance;
}

XRSLAMManager::XRSLAMManager() {}
XRSLAMManager::~XRSLAMManager() {}

static const unsigned char logo_ascii[] = {
    0x0A, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0x20,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x95, 0x97, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0xE2, 0x96,
    0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x95, 0x97, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x0A, 0xE2, 0x95, 0x9A,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x9D, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90,
    0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0x20, 0x20, 0x20, 0x20, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x97, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x0A, 0x20, 0xE2, 0x95, 0x9A, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2,
    0x95, 0x9D, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94,
    0xE2, 0x95, 0x9D, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91,
    0x20, 0x20, 0x20, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x94, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x91, 0x0A, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95,
    0x94, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0x20, 0x20, 0x20,
    0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95,
    0x90, 0xE2, 0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95,
    0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x95,
    0x9A, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95,
    0x9D, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x0A, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x9D, 0x20,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x91, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95,
    0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0x20, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x0A, 0xE2, 0x95, 0x9A, 0xE2, 0x95,
    0x90, 0xE2, 0x95, 0x9D, 0x20, 0x20, 0xE2, 0x95, 0x9A, 0xE2, 0x95, 0x90,
    0xE2, 0x95, 0x9D, 0xE2, 0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D,
    0x20, 0x20, 0xE2, 0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0x20, 0x20, 0xE2, 0x95,
    0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2, 0x95, 0x9A, 0xE2, 0x95,
    0x90, 0xE2, 0x95, 0x9D, 0x20, 0x20, 0x20, 0x20, 0x20, 0xE2, 0x95, 0x9A,
    0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D};

void XRSLAMManager::Init(std::shared_ptr<Config> config) {
    std::shared_ptr<PlaceDescriptorExtractor>
        place_descriptor_extractor;
    std::shared_ptr<PlaceDatabase> place_database;
    std::shared_ptr<LocalDescriptorExtractor>
        local_descriptor_extractor;

    const bool retrieval_enabled = relocalization_v1_profile_enabled();
    const bool descriptor_enabled = retrieval_enabled;

    if (descriptor_enabled) {
#if defined(XRSLAM_HAS_EIGENPLACES)
        const char *model_path =
            std::getenv("XRSLAM_EIGENPLACES_MODEL");
        if (!model_path || model_path[0] == '\0') {
            throw std::runtime_error(
                "XRSLAM_RELOCALIZATION_PROFILE=v1 requires "
                "XRSLAM_EIGENPLACES_MODEL");
        }

        place_descriptor_extractor =
            std::make_shared<
                xrslam::extra::EigenPlacesDescriptorExtractor>(
                model_path, false);
#else
        throw std::runtime_error(
            "XRSLAM_RELOCALIZATION_PROFILE=v1 requires a build "
            "configured with XRSLAM_ENABLE_EIGENPLACES=ON");
#endif
    }

    if (retrieval_enabled) {
#if defined(XRSLAM_HAS_USEARCH)
        place_database =
            xrslam::extra::make_usearch_place_database();
#else
        throw std::runtime_error(
            "XRSLAM_RELOCALIZATION_PROFILE=v1 requires a build "
            "configured with XRSLAM_ENABLE_USEARCH=ON");
#endif
    }

    if (relocalization_v1_profile_enabled()) {
        local_descriptor_extractor =
            std::make_shared<
                xrslam::extra::OrbLocalDescriptorExtractor>();
    }

    const char *dense_enable = std::getenv("XRSLAM_DENSE_SEMANTICS_ENABLE");
    const bool dense_requested = dense_enable &&
        std::string(dense_enable) == "1";
    const char *semantic_enabled_check = std::getenv("XRSLAM_SEMANTICS_ENABLE");
    if (dense_requested && (!semantic_enabled_check ||
        std::string(semantic_enabled_check) != "1"))
        throw std::runtime_error("dense semantics require XRSLAM_SEMANTICS_ENABLE=1");
    const char *a2env = std::getenv("XRSLAM_ASYNC_DEPTH_STAGE_A1");
    async_depth_enabled_ = config->has_depth() && !config->has_imu() &&
                           a2env && std::string(a2env) == "1";
    async_depth_stats_ = {};
    delayed_depth_queue_.clear();
    geometry_depth_queue_.clear();
    scale_frame_medians_.clear();
    candidate_frame_medians_.clear();
    detail_ = std::make_unique<XRSLAM::Detail>(config);
    const char *semantic_enable = std::getenv("XRSLAM_SEMANTICS_ENABLE");
    if (semantic_enable && std::string(semantic_enable) == "1") {
#if defined(XRSLAM_HAS_PIDNET)
        const char *model = std::getenv("XRSLAM_PIDNET_MODEL");
        if (!model || model[0] == '\0')
            throw std::runtime_error(
                "XRSLAM_SEMANTICS_ENABLE=1 requires XRSLAM_PIDNET_MODEL");
        const char *device = std::getenv("XRSLAM_PIDNET_DEVICE");
        const char *outdir = std::getenv("XRSLAM_SEMANTIC_DUMP_DIR");
        const char *voxel_env = std::getenv("XRSLAM_DENSE_VOXEL_SIZE_M");
        const char *stride_env = std::getenv("XRSLAM_DENSE_PIXEL_STRIDE");
        const double voxel = voxel_env ? std::stod(voxel_env) : 0.03;
        const int stride = stride_env ? std::stoi(stride_env) : 4;
        DenseSemanticQualityOptions quality;
        const char *quality_enable = std::getenv("XRSLAM_DENSE_QUALITY_ENABLE");
        quality.enabled = quality_enable && std::string(quality_enable) == "1";
        if (quality.enabled) {
            if (!dense_requested)
                throw std::invalid_argument(
                    "dense quality requires XRSLAM_DENSE_SEMANTICS_ENABLE=1");
            const char *abs_m = std::getenv("XRSLAM_DENSE_EDGE_ABS_M");
            const char *relative = std::getenv("XRSLAM_DENSE_EDGE_REL");
            const char *views = std::getenv("XRSLAM_DENSE_MIN_VIEWS");
            const char *neighbors = std::getenv("XRSLAM_DENSE_MIN_NEIGHBORS");
            quality.depth_edge_abs_m = abs_m ? std::stod(abs_m) : 0.08;
            quality.depth_edge_relative = relative ? std::stod(relative) : 0.03;
            const int minimum_views = views ? std::stoi(views) : 2;
            const int minimum_neighbors = neighbors ? std::stoi(neighbors) : 2;
            if (minimum_views < 1 || minimum_views > 64 ||
                minimum_neighbors < 0 || minimum_neighbors > 26)
                throw std::invalid_argument(
                    "dense quality view/neighbor bounds exceeded");
            quality.min_views = static_cast<std::uint32_t>(minimum_views);
            quality.min_neighbors = static_cast<std::uint32_t>(minimum_neighbors);
            quality.validate();
            std::fprintf(stderr,
                "[DenseSemanticQuality] event=enabled edge_abs_m=%.5f "
                "edge_rel=%.5f min_views=%u min_neighbors=%u "
                "depth_holes_rejected=1 output_only=1\n",
                quality.depth_edge_abs_m, quality.depth_edge_relative,
                quality.min_views, quality.min_neighbors);
        }
        // Stable, explicit class names and model version; a new indoor
        // checkpoint can never silently reinterpret old numeric labels.
        const char *taxonomy_env=std::getenv("XRSLAM_PIDNET_TAXONOMY");
        const std::string taxonomy=taxonomy_env && taxonomy_env[0] ?
            taxonomy_env : "cityscapes19";
        const char *model_id_env=std::getenv("XRSLAM_PIDNET_MODEL_ID");
        const std::string model_id=model_id_env && model_id_env[0] ?
            model_id_env : std::string(model);
        std::vector<std::string> class_names;
        if(const char *raw=std::getenv("XRSLAM_PIDNET_CLASS_NAMES")) {
            std::stringstream ss(raw);std::string s;
            while(std::getline(ss,s,','))class_names.push_back(s);
        } else if(taxonomy=="cityscapes19") {
            class_names={"road","sidewalk","building","wall","fence","pole",
                "traffic light","traffic sign","vegetation","terrain","sky",
                "person","rider","car","truck","bus","train","motorcycle","bicycle"};
        }
        std::uint32_t class_count=19;
        if(const char *raw=std::getenv("XRSLAM_PIDNET_NUM_CLASSES")) {
            const long parsed=std::strtol(raw,nullptr,10);
            if(parsed<1 || parsed>255)throw std::runtime_error("invalid semantic class count");
            class_count=static_cast<std::uint32_t>(parsed);
        }
        if(class_names.size()!=class_count)
            throw std::runtime_error("XRSLAM_PIDNET_CLASS_NAMES must enumerate every model class");
        detail_->set_semantic_keyframe_sink(
            std::make_shared<xrslam::extra::PidNetSemanticWorker>(
                model, device && device[0] ? device : "cpu", class_count,
                1024, 512, taxonomy, outdir ? outdir : "", 2,
                dense_requested, voxel, stride, quality,model_id,class_names));
        if (dense_requested)
            std::fprintf(stderr, "[DenseSemantic] event=enabled voxel=%.6f "
                                 "pixel_stride=%d max_voxels=250000\n",
                         voxel, stride);
        std::fprintf(stderr,
            "[PIDNetSemantic] event=enabled device=%s classes=19 "
            "canvas=1024x512 queue=2 state_mutation=0\n",
            device && device[0] ? device : "cpu");
#else
        throw std::runtime_error(
            "XRSLAM_SEMANTICS_ENABLE=1 requires XRSLAM_ENABLE_PIDNET=ON");
#endif
    }
    detail_->set_place_descriptor_extractor(
        std::move(place_descriptor_extractor));
    detail_->set_place_database(
        std::move(place_database));
    detail_->set_local_descriptor_extractor(
        std::move(local_descriptor_extractor));
    config_ = config;
    log_message(XRSLAM_LOG_INFO, (char *)logo_ascii, XRSLAM_VERSION_STRING);
    config_->log_config();
    std::cout << "-----------------Create XRSLAM v" << XRSLAM_VERSION
              << " successfully-----------" << std::endl;
}

void XRSLAMManager::Destroy() {
    // Explicitly stop the tracking threads and join the asynchronous PIDNet
    // worker while the runtime libraries are still alive. The previous
    // implementation deferred Detail destruction until singleton teardown.
    detail_.reset();
    std::cout << "-----------------Destroy XRSLAM v" << XRSLAM_VERSION
              << " successfully-----------" << std::endl;
}

size_t XRSLAMManager::CopySemanticMap(
    XRSLAMSemanticMapPoint *dst, size_t capacity,
    size_t *total, bool stable_only) const {
    if (total) *total = 0;
    if (capacity && !dst) return 0;
#if defined(XRSLAM_HAS_PIDNET)
    if (!detail_) return 0;
    // The configured PIDNet sink is held alive by Detail. The worker returns
    // immutable value copies under its own mutex, never pointers to its map.
    auto *worker = dynamic_cast<extra::PidNetSemanticWorker *>(
        detail_->semantic_keyframe_sink());
    if (!worker) return 0;
    const auto snapshot = worker->landmark_snapshot(std::nullopt, stable_only);
    if (total) *total = snapshot.size();
    const size_t copied = std::min(capacity, snapshot.size());
    for (size_t i = 0; i < copied; ++i) {
        const auto &p = snapshot[i];
        auto &out = dst[i];
        out.session_id = p.session_id;
        out.track_id = static_cast<uint64_t>(p.track_id);
        out.latest_frame_id = p.latest_frame_id;
        for (int j = 0; j < 3; ++j)
            out.xyz[j] = p.position_session_world[j];
        out.consensus = p.consensus_fraction;
        out.mean_pixel_confidence = p.mean_pixel_confidence;
        out.observation_count = p.observation_count;
        out.winning_label_views = p.winning_label_views;
        out.semantic_class = p.winning_label;
        out.stable = p.stable ? 1 : 0;
    }
    return copied;
#else
    (void)dst;
    (void)capacity;
    (void)stable_only;
    return 0;
#endif
}

size_t XRSLAMManager::QuerySemanticKeyframes(
    const char *class_name, double min_mean_confidence,
    double min_pixel_coverage, size_t max_results,
    XRSLAMSemanticKeyframeHit *dst, size_t capacity, size_t *total) const {
    if(total)*total=0;
    if((capacity && !dst)||!class_name||!class_name[0])return 0;
#if defined(XRSLAM_HAS_PIDNET)
    if(!detail_)return 0;
    auto *worker=dynamic_cast<extra::PidNetSemanticWorker*>(
        detail_->semantic_keyframe_sink());
    if(!worker)return 0;
    const auto hits=worker->query_class(class_name,min_mean_confidence,
                                      min_pixel_coverage,max_results);
    if(total)*total=hits.size();
    const auto count=std::min(capacity,hits.size());
    for(size_t i=0;i<count;++i) {
        const auto &h=hits[i];auto &o=dst[i];o={};
        o.session_id=h.session_id;o.frame_id=h.frame_id;
        o.timestamp=h.timestamp;o.class_id=h.class_id;
        o.coverage=h.coverage;o.mean_confidence=h.mean_confidence;
        for(int k=0;k<3;++k)o.body_position[k]=h.body_position[k];
        for(int k=0;k<4;++k)o.body_quaternion_xyzw[k]=h.body_quaternion_xyzw[k];
        o.map_aligned=h.map_aligned?1:0;
        std::snprintf(o.taxonomy,sizeof(o.taxonomy),"%s",h.taxonomy.c_str());
        std::snprintf(o.model_id,sizeof(o.model_id),"%s",h.model_id.c_str());
    }
    return count;
#else
    (void)min_mean_confidence;(void)min_pixel_coverage;(void)max_results;
    (void)dst;return 0;
#endif
}

bool XRSLAMManager::WaitForSemanticIdle() const {
#if defined(XRSLAM_HAS_PIDNET)
    if (!detail_) return false;
    auto *worker = dynamic_cast<extra::PidNetSemanticWorker *>(
        detail_->semantic_keyframe_sink());
    if (!worker) return false;
    worker->wait_until_idle();
    return true;
#else
    return false;
#endif
}

size_t XRSLAMManager::CopyDenseSemanticMap(
    XRSLAMDenseSemanticPoint *dst, size_t capacity,
    size_t *total, bool stable_only) const {
    if (total) *total = 0;
    if (capacity && !dst) return 0;
#if defined(XRSLAM_HAS_PIDNET)
    if (!detail_) return 0;
    auto *worker = dynamic_cast<extra::PidNetSemanticWorker *>(
        detail_->semantic_keyframe_sink());
    if (!worker) return 0;
    const auto points = worker->dense_snapshot(stable_only);
    if (total) *total = points.size();
    const size_t copied = std::min(capacity, points.size());
    for (size_t i = 0; i < copied; ++i) {
        const auto &p = points[i];
        auto &d = dst[i];
        d.session_id = p.session_id;
        d.voxel_ix = p.ix; d.voxel_iy = p.iy; d.voxel_iz = p.iz;
        for (int k = 0; k < 3; ++k) {
            d.xyz[k] = p.xyz[k];
            d.rgb[k] = p.rgb[k];
        }
        d.semantic_class = p.semantic_class;
        d.consensus = p.consensus;
        d.observation_count = p.observation_count;
        d.stable = p.stable ? 1 : 0;
    }
    return copied;
#else
    return 0;
#endif
}

int XRSLAMManager::CheckLicense(const char *license_path,
                                const char *product_name) {
    return 1;
}

void XRSLAMManager::PushImage(XRSLAMImage *image) {
    // left img
    if (image->camera_id == 0) {
        std::shared_ptr<xrslam::extra::OpenCvImage> opencv_image =
            std::make_shared<xrslam::extra::OpenCvImage>();
        int cols = config_->camera_resolution()[0];
        int rows = config_->camera_resolution()[1];
        opencv_image->t = image->timeStamp;

        cv::Mat img;
        if(image->channel == 1){
            img = cv::Mat(rows, cols, CV_8UC1, image->data, image->stride);
            opencv_image->image = img.clone();
        }
        else if(image->channel == 3){
            img = cv::Mat(rows, cols, CV_8UC3, image->data, image->stride);
            cv::cvtColor(img, opencv_image->image, cv::COLOR_BGR2GRAY);
        }
        else if(image->channel == 4){
            img = cv::Mat(rows, cols, CV_8UC4, image->data, image->stride);
            cv::cvtColor(img, opencv_image->image, cv::COLOR_BGRA2GRAY);
        }
        else{
            std::cerr << "Image channel is not supported!" << std::endl;
            exit(-1);
        }
            
        // Preserve the canonical pre-CLAHE camera image for place recognition.
        // The OpenCvImage source initially shares this allocation, so enabling
        // global/local descriptors does not add another full-frame copy.
        // If correct_distortion() is used, both the frontend image and this
        // source are rectified into the same pixel geometry.
        opencv_image->raw = img.clone();
        opencv_image->set_place_recognition_source(opencv_image->raw);

        std::lock_guard<std::mutex> lck(input_mutex_);
        if (!pending_depth_.empty() &&
            std::abs(pending_depth_timestamp_ - image->timeStamp) < 1.0e-6 &&
            pending_depth_.cols == cols && pending_depth_.rows == rows) {
            opencv_image->depth_image = pending_depth_;
            opencv_image->depth_source_type = pending_depth_source_;
        }
        pending_depth_.release();
        pending_depth_timestamp_ = -1.0;
        cur_image_ = std::shared_ptr<xrslam::Image>(opencv_image);
    }
}

void XRSLAMManager::PushDepth(XRSLAMDepthImage *depth) {
    if (!depth || depth->width <= 0 || depth->height <= 0 ||
        depth->stride <= 0)
        return;

    cv::Mat meters;
    if (depth->format == XRSLAM_DEPTH_FLOAT32) {
        if (!depth->data_f32)
            return;
        cv::Mat raw(depth->height, depth->width, CV_32FC1, depth->data_f32,
                    (size_t)depth->stride);
        meters = raw.clone();
    } else {
        if (!depth->data || depth->scale <= 0.0)
            return;
        cv::Mat raw(depth->height, depth->width, CV_16UC1, depth->data,
                    (size_t)depth->stride);
        raw.convertTo(meters, CV_32FC1, depth->scale);
    }

    DepthSource source = DepthSource::SENSOR_METRIC;
    if (depth->source == XRSLAM_DEPTH_MONOCULAR_METRIC)
        source = DepthSource::MONOCULAR_METRIC;
    else if (depth->source == XRSLAM_DEPTH_MONOCULAR_RELATIVE)
        source = DepthSource::MONOCULAR_RELATIVE;

    std::lock_guard<std::mutex> lck(input_mutex_);
    if (async_depth_enabled_) {
        ++async_depth_stats_.received;
        if (!std::isfinite(depth->timeStamp) ||
            meters.cols != config_->camera_resolution()[0] ||
            meters.rows != config_->camera_resolution()[1]) {
            ++async_depth_stats_.rejected;
            return;
        }
        // Bounded, owned copies; no frame/map pointer survives the API call.
        if (delayed_depth_queue_.size() == 8) {
            delayed_depth_queue_.pop_front();
            ++async_depth_stats_.rejected;
        }
        delayed_depth_queue_.push_back(
            DelayedDepth{depth->timeStamp, std::move(meters), source});
        async_depth_stats_.pending = delayed_depth_queue_.size();
        return;
    }
    pending_depth_ = meters;
    pending_depth_timestamp_ = depth->timeStamp;
    pending_depth_source_ = source;
}

void XRSLAMManager::PushAcceleration(XRSLAMAcceleration *acc) {
    detail_->track_accelerometer(acc->timestamp, acc->data[0], acc->data[1],
                                 acc->data[2]);
}

void XRSLAMManager::PushGyroscope(XRSLAMGyroscope *gyro) {
    detail_->track_gyroscope(gyro->timestamp, gyro->data[0], gyro->data[1],
                             gyro->data[2]);
}

void XRSLAMManager::RunOneFrame() {
    std::lock_guard<std::mutex> lck(input_mutex_);
    detail_->track_camera(cur_image_);
    if (async_depth_enabled_) ProcessDelayedDepthLocked();
}

bool XRSLAMManager::GetAsyncDepthStats(XRSLAMAsyncDepthStats *out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lck(input_mutex_);
    *out = async_depth_stats_;
    if (detail_ && detail_->frontend)
        out->optimized_snapshots =
            detail_->frontend->optimized_depth_snapshot_count();
    return true;
}

namespace {
double median_of(std::vector<double> &v) {
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const double upper = v[mid];
    if (v.size() % 2 != 0) return upper;
    std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
    return 0.5 * (upper + v[mid - 1]);
}
} // namespace

void XRSLAMManager::ProcessDelayedDepthLocked() {
    // Stage A2 RGB-frame association remains separate from the optimized
    // Stage A3 geometry queue. Never keep a live Frame/Track pointer.
    if (!detail_ || !detail_->feature_tracker ||
        !detail_->feature_tracker->map) return;
    auto *map = detail_->feature_tracker->map.get();
    {
        auto guard = map->lock();
        if (map->frame_num() != 0) {
            constexpr double epsilon = 1.0e-6;
            const double oldest = map->get_frame(0)->image->t;
            const double newest =
                map->get_frame(map->frame_num() - 1)->image->t;
            std::deque<DelayedDepth> waiting;
            while (!delayed_depth_queue_.empty()) {
                DelayedDepth sample = std::move(delayed_depth_queue_.front());
                delayed_depth_queue_.pop_front();
                if (sample.timestamp < oldest - epsilon) {
                    ++async_depth_stats_.too_old;
                    continue;
                }
                if (sample.timestamp > newest + epsilon) {
                    waiting.push_back(std::move(sample));
                    continue;
                }
                const Frame *matched = nullptr;
                for (size_t i = 0; i < map->frame_num(); ++i) {
                    const Frame *candidate = map->get_frame(i);
                    if (std::abs(candidate->image->t - sample.timestamp) <
                        epsilon) {
                        matched = candidate;
                        break;
                    }
                }
                if (!matched) {
                    ++async_depth_stats_.unknown;
                    continue;
                }
                ++async_depth_stats_.associated;
                std::vector<double> valid_depths;
                for (size_t i = 0; i < matched->keypoint_num(); ++i) {
                    if (!matched->get_track(i)) continue;
                    ++async_depth_stats_.tracked_feature_samples;
                    const auto pixel =
                        apply_k(matched->get_keypoint(i), matched->K);
                    if (!pixel.allFinite()) continue;
                    const int x = static_cast<int>(std::lround(pixel(0)));
                    const int y = static_cast<int>(std::lround(pixel(1)));
                    if (x < 0 || y < 0 || x >= sample.meters.cols ||
                        y >= sample.meters.rows) continue;
                    const float z = sample.meters.at<float>(y, x);
                    if (!std::isfinite(z) || z <= 0.05f || z >= 50.0f)
                        continue;
                    ++async_depth_stats_.valid_feature_samples;
                    valid_depths.push_back(z);
                    if (async_depth_stats_.depth_min_m == 0.0 ||
                        z < async_depth_stats_.depth_min_m)
                        async_depth_stats_.depth_min_m = z;
                    async_depth_stats_.depth_max_m =
                        std::max(async_depth_stats_.depth_max_m,
                                 static_cast<double>(z));
                }
                if (!valid_depths.empty())
                    async_depth_stats_.latest_depth_median_m =
                        median_of(valid_depths);

                // Transfer the owned depth image, not a pointer to the
                // continuous tracker. Optimized snapshots arrive later.
                constexpr size_t max_geometry_queue = 16;
                if (geometry_depth_queue_.size() >= max_geometry_queue) {
                    geometry_depth_queue_.pop_front();
                    ++async_depth_stats_.geometry_frames_expired;
                    ++async_depth_stats_.geometry_queue_evictions;
                }
                geometry_depth_queue_.push_back(std::move(sample));
            }
            delayed_depth_queue_.swap(waiting);
        }
        async_depth_stats_.pending = delayed_depth_queue_.size();
    }
    // No FeatureTracker map lock is held while obtaining frontend snapshots.
    ProcessOptimizedDepthLocked();
}

void XRSLAMManager::ProcessOptimizedDepthLocked() {
    if (!detail_ || !detail_->frontend || !cur_image_) return;
    const double latest_rgb_timestamp = cur_image_->t;
    if (!std::isfinite(latest_rgb_timestamp)) return;
    async_depth_stats_.latest_rgb_timestamp_s = latest_rgb_timestamp;
    constexpr double max_wait_seconds = 6.0;
    std::deque<DelayedDepth> waiting;
    while (!geometry_depth_queue_.empty()) {
        DelayedDepth sample = std::move(geometry_depth_queue_.front());
        geometry_depth_queue_.pop_front();

        OptimizedDepthSnapshot snapshot;
        if (!detail_->frontend->copy_optimized_depth_snapshot(
                sample.timestamp, snapshot)) {
            if (latest_rgb_timestamp - sample.timestamp >
                max_wait_seconds) {
                // Not selected for the recent optimized history; this is
                // expected for many RGB frames and is not a sensor error.
                ++async_depth_stats_.geometry_frames_unselected;
            } else {
                ++async_depth_stats_.geometry_missing_snapshot_waits;
                waiting.push_back(std::move(sample));
            }
            continue;
        }
        // A snapshot without triangulated observations may be replaced by a
        // later optimized version, so allow it to mature until expiry.
        if (snapshot.points.empty()) {
            if (latest_rgb_timestamp - sample.timestamp >
                max_wait_seconds) {
                ++async_depth_stats_.geometry_frames_expired;
                ++async_depth_stats_.geometry_empty_snapshot_expirations;
            } else {
                ++async_depth_stats_.geometry_empty_snapshot_waits;
                waiting.push_back(std::move(sample));
            }
            continue;
        }
        async_depth_stats_.latest_geometry_matched_timestamp_s = sample.timestamp;
        ++async_depth_stats_.geometry_frames_matched;
        std::vector<double> ratios;
        for (const auto &obs : snapshot.points) {
            if (!std::isfinite(obs.visual_z) || obs.visual_z <= 0.05 ||
                obs.x < 0 || obs.y < 0 ||
                obs.x >= sample.meters.cols ||
                obs.y >= sample.meters.rows)
                continue;
            const float depth = sample.meters.at<float>(obs.y, obs.x);
            if (!std::isfinite(depth) || depth <= 0.05f ||
                depth >= 50.0f)
                continue;
            ++async_depth_stats_.geometry_samples;
            const double ratio = static_cast<double>(depth) / obs.visual_z;
            if (!std::isfinite(ratio) || ratio <= 0.0) continue;
            ratios.push_back(ratio);
            ++async_depth_stats_.ratio_samples;
        }
        const size_t usable_ratio_count = ratios.size();
        if (usable_ratio_count == 0)
            ++async_depth_stats_.geometry_frames_zero_ratios;
        else if (usable_ratio_count < 4)
            ++async_depth_stats_.geometry_frames_1_to_3_ratios;
        else if (usable_ratio_count < 10)
            ++async_depth_stats_.geometry_frames_4_to_9_ratios;
        else
            ++async_depth_stats_.geometry_frames_10_plus_ratios;

        // A4.1: independent low-density candidate estimator. Do not alter
        // Stage A3 ratio_frames, per-frame median or 32-frame statistics.
        // Pixels outside a factor-4 envelope around their own frame median
        // are obvious disagreements; remaining variability is measured.
        if (usable_ratio_count >= 4) {
            std::vector<double> raw_ratios = ratios;
            const double raw_median = median_of(raw_ratios);
            std::vector<double> trimmed;
            trimmed.reserve(ratios.size());
            for (double value : ratios) {
                if (value >= raw_median * 0.25 &&
                    value <= raw_median * 4.0)
                    trimmed.push_back(value);
            }
            if (trimmed.size() >= 4) {
                const double candidate_med = median_of(trimmed);
                std::vector<double> differences;
                differences.reserve(trimmed.size());
                for (double value : trimmed)
                    differences.push_back(std::abs(value - candidate_med));
                const double candidate_mad = median_of(differences);
                if (std::isfinite(candidate_med) &&
                    candidate_med >= 0.001 && candidate_med <= 100.0 &&
                    candidate_mad / candidate_med <= 0.75) {
                    // A large scale jump is a different visual gauge or a
                    // gross inconsistency, not an inlier to average.
                    // Start a fresh consecutive-candidate segment.
                    if (!candidate_frame_medians_.empty()) {
                        std::vector<double> history(
                            candidate_frame_medians_.begin(),
                            candidate_frame_medians_.end());
                        const double prior = median_of(history);
                        if (candidate_med < prior / 2.0 ||
                            candidate_med > prior * 2.0) {
                            candidate_frame_medians_.clear();
                            ++async_depth_stats_.candidate_scale_resets;
                        }
                    }
                    candidate_frame_medians_.push_back(candidate_med);
                    constexpr size_t candidate_capacity = 16;
                    if (candidate_frame_medians_.size() > candidate_capacity)
                        candidate_frame_medians_.pop_front();
                    std::vector<double> window(
                        candidate_frame_medians_.begin(),
                        candidate_frame_medians_.end());
                    const double window_median = median_of(window);
                    for (double &value : window)
                        value = std::abs(value - window_median);
                    const double window_mad = median_of(window);
                    async_depth_stats_.candidate_window_frames =
                        candidate_frame_medians_.size();
                    async_depth_stats_.candidate_window_median =
                        window_median;
                    async_depth_stats_.candidate_window_mad =
                        window_mad;
                    async_depth_stats_.candidate_window_relative_mad =
                        window_median > 0.0 ? window_mad / window_median : 0.0;
                    ++async_depth_stats_.candidate_frames_accepted;
                } else {
                    ++async_depth_stats_.candidate_frames_rejected_noisy;
                }
            } else {
                ++async_depth_stats_.candidate_frames_rejected_noisy;
            }
        }

        if (ratios.size() >= 10) {
            const double med = median_of(ratios);
            std::vector<double> deviations;
            deviations.reserve(ratios.size());
            for (double ratio : ratios)
                deviations.push_back(std::abs(ratio - med));
            async_depth_stats_.latest_ratio_median = med;
            async_depth_stats_.latest_ratio_mad = median_of(deviations);
            ++async_depth_stats_.ratio_frames;

            // Median of FRAME medians avoids a high-feature-count frame
            // dominating the estimate. Relative MAD reports consistency,
            // never grants metric authority by itself.
            scale_frame_medians_.push_back(med);
            constexpr size_t max_scale_frames = 32;
            if (scale_frame_medians_.size() > max_scale_frames)
                scale_frame_medians_.pop_front();
            std::vector<double> window(scale_frame_medians_.begin(),
                                       scale_frame_medians_.end());
            const double window_med = median_of(window);
            for (double &value : window)
                value = std::abs(value - window_med);
            const double window_mad = median_of(window);
            async_depth_stats_.scale_window_frames =
                scale_frame_medians_.size();
            async_depth_stats_.scale_window_median = window_med;
            async_depth_stats_.scale_window_mad = window_mad;
            async_depth_stats_.scale_window_relative_mad =
                window_med > 0.0 ? window_mad / window_med : 0.0;
        }
        // Read-only. No metric constraints, inverse depth or pose writes.
    }
    geometry_depth_queue_.swap(waiting);
    async_depth_stats_.geometry_pending = geometry_depth_queue_.size();
}

void XRSLAMManager::GetResultBodyPose(XRSLAMPose *pose) const {
    std::tuple<double, Pose> latest_state = detail_->get_latest_pose();
    pose->timestamp = std::get<0>(latest_state);
    Pose latest_pose = std::get<1>(latest_state);
    Pose body_pose;
    body_pose.q = latest_pose.q * config_->imu_to_body_rotation();
    body_pose.p = latest_pose.p + latest_pose.q * config_->imu_to_body_translation();
    for (int i = 0; i < 3; i++)
        pose->translation[i] = body_pose.p(i);
    pose->quaternion[0] = body_pose.q.x();
    pose->quaternion[1] = body_pose.q.y();
    pose->quaternion[2] = body_pose.q.z();
    pose->quaternion[3] = body_pose.q.w();
}

bool XRSLAMManager::GetVerifiedMapAlignment(
    XRSLAMVerifiedMapAlignment *out) const {
    if (!out || !detail_) return false;
    *out = {};
    const auto s = detail_->get_verified_map_localization();
    out->valid = s.valid ? 1 : 0;
    out->reference_map_loaded = s.reference_map_loaded ? 1 : 0;
    out->tracking_session_epoch = s.tracking_session_epoch;
    out->reference_session_id = s.reference_session_id;
    out->verified_timestamp = s.verified_timestamp;
    out->yaw_rad = s.T_map_vio.yaw_rad;
    for (int i=0;i<3;++i)
        out->translation[i] = s.T_map_vio.translation[i];
    std::snprintf(out->map_id,sizeof(out->map_id),"%s",s.map_id.c_str());
    return true;
}

void XRSLAMManager::GetResultGlobalBodyPose(XRSLAMPose *pose) const {
    // Copy the existing body output; never mutate the estimator/local result.
    GetResultBodyPose(pose);
    PlaceGraph4DoFCorrection correction;
    if (pose->timestamp <= 0.0 ||
        !detail_->get_latest_global_correction(correction) ||
        pose->timestamp < correction.timestamp ||
        !std::isfinite(correction.yaw_deg) ||
        !correction.translation.allFinite()) {
        return;
    }

    const vector<3> local_p(
        pose->translation[0], pose->translation[1], pose->translation[2]);
    const quaternion local_q(
        pose->quaternion[3], pose->quaternion[0],
        pose->quaternion[1], pose->quaternion[2]);
    if (!local_p.allFinite() || !local_q.coeffs().allFinite() ||
        local_q.squaredNorm() < 1.0e-20) {
        return;
    }

    constexpr double radians_per_degree =
        3.14159265358979323846 / 180.0;
    const quaternion drift_q(Eigen::AngleAxisd(
        correction.yaw_deg * radians_per_degree,
        Eigen::Vector3d::UnitZ()));
    const vector<3> global_p = drift_q * local_p + correction.translation;
    const quaternion global_q = (drift_q * local_q).normalized();
    pose->translation[0] = global_p.x();
    pose->translation[1] = global_p.y();
    pose->translation[2] = global_p.z();
    pose->quaternion[0] = global_q.x();
    pose->quaternion[1] = global_q.y();
    pose->quaternion[2] = global_q.z();
    pose->quaternion[3] = global_q.w();
}

void XRSLAMManager::GetResultCameraPose(XRSLAMPose *pose) const {
    std::tuple<double, Pose> latest_state = detail_->get_latest_pose();
    pose->timestamp = std::get<0>(latest_state);
    Pose latest_pose = std::get<1>(latest_state);
    Pose camera_pose;
    camera_pose.q = latest_pose.q * config_->camera_to_body_rotation();
    camera_pose.p = latest_pose.p + latest_pose.q * config_->camera_to_body_translation();
    for (int i = 0; i < 3; i++)
        pose->translation[i] = camera_pose.p(i);
    pose->quaternion[0] = camera_pose.q.x();
    pose->quaternion[1] = camera_pose.q.y();
    pose->quaternion[2] = camera_pose.q.z();
    pose->quaternion[3] = camera_pose.q.w();
}

void XRSLAMManager::GetInfoIntrinsics(XRSLAMIntrinsics *intrinsics) const {
    intrinsics->fx = config_->camera_intrinsic()(0, 0);
    intrinsics->fy = config_->camera_intrinsic()(1, 1);
    intrinsics->cx = config_->camera_intrinsic()(0, 2);
    intrinsics->cy = config_->camera_intrinsic()(1, 2);
}


void XRSLAMManager::GetResultState(XRSLAMState *state) const {
    SysState cur_state = detail_->get_system_state();
    if (cur_state == SYS_INITIALIZING) {
        *state = XRSLAM_STATE_INITIALIZING;
    } else if (cur_state == SYS_TRACKING) {
        *state = XRSLAM_STATE_TRACKING_SUCCESS;
    } else if (cur_state == SYS_CRASH) {
        *state = XRSLAM_STATE_TRACKING_FAIL;
    } else if (cur_state == SYS_UNKNOWN) {
        *state = XRSLAM_STATE_TRACKING_FAIL;
    }
}
void XRSLAMManager::GetResultLandmarks(XRSLAMLandmarks *landmarks) const {
    if (!landmarks) return;
    landmarks->num_landmarks = 0;
    landmarks->landmarks = nullptr;
    inspect(sliding_window_landmarks, swlandmarks) {
        if (!swlandmarks.has_value()) return;
        auto pts = std::any_cast<std::vector<xrslam::Landmark>>(swlandmarks);
        if (pts.empty()) return;
        landmarks->num_landmarks = static_cast<int>(pts.size());
        landmarks->landmarks = new XRSLAMLandmark[pts.size()];
        for (int i = 0; i < pts.size(); i++) {
            xrslam::vector<3> cur_p = pts[i].p;
            landmarks->landmarks[i].x = cur_p(0);
            landmarks->landmarks[i].y = cur_p(1);
            landmarks->landmarks[i].z = cur_p(2);
            landmarks->landmarks[i].track_id = pts[i].track_id;
            landmarks->landmarks[i].triangulated = pts[i].triangulated ? 1 : 0;
        }
    }
}

void XRSLAMManager::GetResultFeatures(XRSLAMFeatures *features) const {
    if (!features) return;
    features->pos.clear();
    inspect(feature_tracker_tracked_pixels, tracked_pixels) {
        if (tracked_pixels.has_value()) {
            auto pixels = std::any_cast<std::vector<std::pair<double, double>>>(
                tracked_pixels);
            features->pos.reserve(pixels.size());
            for (const auto &p : pixels)
                features->pos.push_back({p.first, p.second});
        }
    }
}

void XRSLAMManager::GetResultBias(XRSLAMIMUBias *bias) const {
    inspect_debug(sliding_window_current_bg, bg) {
        if (bg.has_value()) {
            xrslam::vector<3> gyr_bias = std::any_cast<xrslam::vector<3>>(bg);
            bias->gyr_bias.data[0] = gyr_bias(0);
            bias->gyr_bias.data[1] = gyr_bias(1);
            bias->gyr_bias.data[2] = gyr_bias(2);
        }
    }
    inspect_debug(sliding_window_current_ba, ba) {
        if (ba.has_value()) {
            xrslam::vector<3> acc_bias = std::any_cast<xrslam::vector<3>>(ba);
            bias->acc_bias.data[0] = acc_bias(0);
            bias->acc_bias.data[1] = acc_bias(1);
            bias->acc_bias.data[2] = acc_bias(2);
        }
    }
}

void XRSLAMManager::GetResultVersion(XRSLAMStringOutput *output) const {
    output->str_length = strlen(XRSLAM_VERSION);
    output->data = new char[output->str_length + 5];
    strcpy(output->data, XRSLAM_VERSION);
}

} // namespace xrslam
