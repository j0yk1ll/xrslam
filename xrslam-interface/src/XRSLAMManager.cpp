#include "XRSLAMManager.h"

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
    inspect_debug(sliding_window_landmarks, swlandmarks) {
        auto pts = std::any_cast<std::vector<xrslam::Landmark>>(swlandmarks);
        landmarks->num_landmarks = pts.size();
        landmarks->landmarks = new XRSLAMLandmark[pts.size()];
        for (int i = 0; i < pts.size(); i++) {
            xrslam::vector<3> cur_p = pts[i].p;
            landmarks->landmarks[i].x = cur_p(0);
            landmarks->landmarks[i].y = cur_p(1);
            landmarks->landmarks[i].z = cur_p(2);
        }
    }
}

void XRSLAMManager::GetResultFeatures(XRSLAMFeatures *features) const {}

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
