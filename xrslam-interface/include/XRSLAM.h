/**
 * @file XRSLAM.h
 * @brief XRSlam API
 */

#ifndef _XRSLAM_H_
#define _XRSLAM_H_
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <vector>
extern "C" {
#endif

/******************************************************************************************
 *                                 XRSLAM sensor data
 ******************************************************************************************/
/**
 * @brief input sensor data type
 */
typedef enum XRSLAMSensorType {
    XRSLAM_SENSOR_CAMERA = 0,      /*!< gray image. */
    XRSLAM_SENSOR_DEPTH_CAMERA,    /*!< depth image. */
    XRSLAM_SENSOR_ACCELERATION,    /*!< acceleration. */
    XRSLAM_SENSOR_GYROSCOPE,       /*!< gyroscope. */
    XRSLAM_SENSOR_GRAVITY,         /*!< gravity. */
    XRSLAM_SENSOR_ROTATION_VECTOR, /*!< rotation vector. */
    XRSLAM_SENSOR_UNKNOWN
} XRSLAMSensorType;

/**
 * @brief Image extension info
 */
typedef struct XRSLAMImageExtension {
    double exposure_time;          /*!< image exposure time. */
    double default_focus_distance; /*!< default focus info. */
    double focal_length;           /*!< current focal length. */
    double focus_distance;         /*!< current focus distance. */
} XRSLAMImageExtension;

/**
 * @brief input gray image data
 */
typedef struct XRSLAMImage {
    unsigned char
        *data;        /*!< contains the intensity value for each image pixel. */
    double timeStamp; /*!< timestamp in second. */
    int stride;       /*!< image stride, number of bytes per row. */
    int camera_id;    /*!< camera id. */
    int channel;      /*!< image channel. */
    XRSLAMImageExtension *ext; /*!< ext info of image. */
} XRSLAMImage;

/**
 * @brief input depth image data
 */
typedef enum XRSLAMDepthFormat {
    XRSLAM_DEPTH_UINT16 = 0,
    XRSLAM_DEPTH_FLOAT32 = 1
} XRSLAMDepthFormat;

typedef enum XRSLAMDepthSource {
    XRSLAM_DEPTH_SENSOR_METRIC = 0,
    XRSLAM_DEPTH_MONOCULAR_METRIC = 1,
    XRSLAM_DEPTH_MONOCULAR_RELATIVE = 2
} XRSLAMDepthSource;

typedef struct XRSLAMDepthImage {
    uint16_t *data;       /*!< uint16 depth samples, if format is UINT16. */
    uint16_t *confidence; /*!< optional confidence, may be NULL. */
    double timeStamp;     /*!< timestamp in second. */
    int width;            /*!< depth width in pixels. */
    int height;           /*!< depth height in pixels. */
    int stride;           /*!< row stride in bytes. */
    double scale;         /*!< meters per uint16 unit; ignored for FLOAT32. */
    float *data_f32;      /*!< float32 depth in meters, if format is FLOAT32. */
    XRSLAMDepthFormat format;
    XRSLAMDepthSource source;
} XRSLAMDepthImage;

/**
 * @brief input acceleration data
 */
typedef struct XRSLAMAcceleration {
    double data[3];   /*!< acceleration raw data. */
    double timestamp; /*!< timestamp in second. */
} XRSLAMAcceleration;

/**
 * @brief input gyroscope data
 */
typedef struct XRSLAMGyroscope {
    double data[3];   /*!< gyroscope raw data. */
    double timestamp; /*!< timestamp in second. */
} XRSLAMGyroscope;

/**
 * @brief input gravity direction data
 */
typedef struct XRSLAMGravity {
    double data[3];   /*!< gravity direction. */
    double timestamp; /*!< timestamp in second. */
} XRSLAMGravity;

/**
 * @brief attitude of the device
 */
typedef struct XRSLAMRotationVector {
    double data[4];   /*!< attitude of the device. */
    double timestamp; /*!< timestamp in second. */
} XRSLAMRotationVector;

/******************************************************************************************
 *                                   XRSLAM tracking result
 ******************************************************************************************/
/**
 * @brief slam result data type
 */
typedef enum XRSLAMResultType {
    XRSLAM_RESULT_BODY_POSE = 0, /*!< body pose. */
    XRSLAM_RESULT_CAMERA_POSE,   /*!< camera pose. */
    XRSLAM_RESULT_STATE,         /*!< system state. */
    XRSLAM_RESULT_LANDMARKS,     /*!< 3D landmarks. */
    XRSLAM_RESULT_FEATURES,      /*!< 2D features. */
    XRSLAM_RESULT_BIAS,          /*!< imu bias. */
    XRSLAM_RESULT_DEBUG_LOGS,    /*!< debug logs. */
    XRSLAM_RESULT_VERSION,       /*!< version. */
    XRSLAM_RESULT_UNKNOWN,
    XRSLAM_INFO_INTRINSICS,
    // Output-only 4-DoF place graph correction. Existing values unchanged.
    XRSLAM_RESULT_GLOBAL_BODY_POSE
} XRSLAMResultType;

/**
 * @brief    slam pose.
 * @details  For a 3D point in world coordinate \f$ X_w \f$, its 3D
 *           coordinate in the camera frame \f$ X_c = R * X_w + T \f$
 */
typedef struct XRSLAMPose {
    double quaternion[4];  /*!< quaternion of rotation: format [x, y, z, w]. */
    double translation[3]; /*!< translation vector T. */
    double timestamp;      /*!< timestamp. */
} XRSLAMPose;

typedef struct XRSLAMIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
} XRSLAMIntrinsics;


/**
 * @brief    slam state.
 * @details  SLAM state to show the system status.
 */
typedef enum XRSLAMState {
    XRSLAM_STATE_INITIALIZING,     /*!< SLAM is in initialize state. */
    XRSLAM_STATE_TRACKING_SUCCESS, /*!< SLAM is in tracking state and track
                                      success. */
    XRSLAM_STATE_TRACKING_FAIL /*!< SLAM is in tracking state and track fail. */
} XRSLAMState;

/**
 * @brief  landmark 3d position.
 * @details x, y, z is world position of the point.
 */
typedef struct XRSLAMLandmark {
    double x, y, z;
    /* Native track identity, unique within one XRSLAM tracking session. */
    uint64_t track_id;
    /* Only triangulated landmarks may enter a persistent 3D map. */
    uint8_t triangulated;
} XRSLAMLandmark;
typedef struct XRSLAMLandmarks {
    XRSLAMLandmark *landmarks;
    int num_landmarks;
} XRSLAMLandmarks;

/**
 * @brief  2d corner in image coordinate.
 */
typedef struct XRSLAMFeature {
    double x, y;
} XRSLAMFeature;
typedef struct XRSLAMFeatures {
    struct Point{
        double x; 
        double y;
    };
    std::vector<Point> pos;
} XRSLAMFeatures;

/**
 * @brief IMU bias.
 */
typedef struct XRSLAMBias {
    double data[3];
} XRSLAMBias;
typedef struct XRSLAMIMUBias {
    XRSLAMBias acc_bias;
    XRSLAMBias gyr_bias;
} XRSLAMIMUBias;
/**
 * @brief  debug logs.
 */
typedef struct XRSLAMStringOutput {
    int str_length;
    char *data; /*!< slam information. */
} XRSLAMStringOutput;

/******************************************************************************************
 *                                   XRSLAM function interface
 ******************************************************************************************/

/**
 * @brief create SLAM system with configuration files.
 * @param[in] slam_config_path slam configuration file path.
 * @param[in] device_config_path  device configuration file path, include camera
 * intrinsics, extrinsics and so on.
 * @param[in] license_path  license path
 * @param[in] product_name  product name which user can define
 * @param[out] config configuration of slam and device
 * @return 1 success, otherwise 0
 */
int XRSLAMCreate(const char *slam_config_path, const char *device_config_path,
                 const char *license_path, const char *product_name,
                 void **config);

/**
 * @brief return whether the active configuration requires IMU input.
 * @return 1 when IMU is required, otherwise 0.
 */
int XRSLAMRequiresIMU(void *config);

/**
 * @brief push sensor data to SLAM system
 * @param[in] sensor_type sensor type.
 * @param[in] sensor_data sensor data.
 */
void XRSLAMPushSensorData(XRSLAMSensorType sensor_type, void *sensor_data);

/**
 * @brief end one frame input and run slam
 */
void XRSLAMRunOneFrame();

/* Stage A2: diagnostic-only delayed depth. Counters reset at XRSLAMCreate.
 * No depth priors are added and no metric pose authority is implied. */
typedef struct XRSLAMAsyncDepthStats {
    uint64_t received;
    uint64_t associated;
    uint64_t too_old;
    uint64_t unknown;
    uint64_t rejected;
    uint64_t valid_feature_samples;
    uint64_t tracked_feature_samples;
    uint64_t constraints_added;
    uint64_t pending;
    /* Stage A3: descriptive statistics, not a scale calibration result. */
    uint64_t geometry_samples;   /* finite, positive triangulated camera Z */
    uint64_t ratio_samples;      /* depth prediction / visual camera Z */
    uint64_t ratio_frames;       /* associated frames with >= 10 ratios */
    double depth_min_m;         /* over valid feature samples since create */
    double depth_max_m;
    double latest_depth_median_m; /* last associated frame with valid samples */
    double latest_ratio_median;  /* last frame with >= 10 geometry ratios */
    double latest_ratio_mad;     /* median absolute ratio deviation */
    uint64_t optimized_snapshots;
    uint64_t geometry_frames_matched;
    uint64_t geometry_frames_expired;
    uint64_t geometry_pending;
    /* Stage A3.1: bounded rolling scale statistics (diagnostic only). */
    uint64_t geometry_frames_unselected;
    uint64_t scale_window_frames;
    double scale_window_median;
    double scale_window_mad;
    double scale_window_relative_mad;
    /* A4.1 diagnostic-only candidate selection; independent of A3 counters. */
    uint64_t geometry_frames_zero_ratios;
    uint64_t geometry_frames_1_to_3_ratios;
    uint64_t geometry_frames_4_to_9_ratios;
    uint64_t geometry_frames_10_plus_ratios;
    uint64_t candidate_frames_accepted;
    uint64_t candidate_frames_rejected_noisy;
    uint64_t candidate_scale_resets;
    uint64_t candidate_window_frames;
    double candidate_window_median;
    double candidate_window_mad;
    double candidate_window_relative_mad;
    /* A7.1: read-only optimized-geometry pipeline observability.
     * Queue evictions and empty optimized snapshots are distinct. The
     * timestamp is in the same camera acquisition-clock domain as RGB. */
    uint64_t geometry_queue_evictions;
    uint64_t geometry_empty_snapshot_expirations;
    uint64_t geometry_empty_snapshot_waits;
    uint64_t geometry_missing_snapshot_waits;
    double latest_geometry_matched_timestamp_s;
    double latest_rgb_timestamp_s;
} XRSLAMAsyncDepthStats;
int XRSLAMGetAsyncDepthStats(XRSLAMAsyncDepthStats *out);

void XRSLAMSetViewer(void *viewer);

/**
 * @brief get SLAM results
 * @param[in]  result_type slam result type.
 * @param[out] result_data result data.
 */
void XRSLAMGetResult(XRSLAMResultType result_type, void *result_data);

/* 0122d: read-only semantic map. All XYZ remain in their originating
 * VIO session-local frame. This API does not merge or transform sessions.
 * Caller owns the destination array; no pointer aliases XRSLAM internals.
 * Split track/frame IDs in PLY, but retain full uint64 IDs in this API.
 */
typedef struct XRSLAMSemanticMapPoint {
    uint64_t session_id;
    uint64_t track_id;
    uint64_t latest_frame_id;
    double xyz[3];
    double consensus; /* weighted agreement fraction, not calibrated probability */
    double mean_pixel_confidence;
    uint64_t observation_count;
    uint64_t winning_label_views;
    uint8_t semantic_class;
    uint8_t stable;
} XRSLAMSemanticMapPoint;

/* Returns count actually copied (<=capacity). *total receives the count
 * at the instant of this copy. Query with dst=NULL, capacity=0 first; a
 * concurrent inference may change the total between two calls.
 * stable_only=1 filters to stable (2+ agreeing views, consensus >=0.65).
 * Must not race XRSLAMDestroy or XRSLAMCreate.
 */
size_t XRSLAMGetSemanticMapSnapshot(XRSLAMSemanticMapPoint *dst,
                                    size_t capacity, size_t *total,
                                    int stable_only);

/* Wait for the jobs currently accepted by the semantic worker to finish.
 * Call only after dataset submission ends; do not invoke in the VIO loop.
 * Returns 1 when semantics are enabled, 0 otherwise.
 */
int XRSLAMWaitForSemanticIdle(void);

/* Read-only depth-backed semantic voxel cloud, in session-local coordinates.
 * RGB is canonical camera appearance; class is PIDNet semantic index.
 * No session merging, TSDF, or mesh connectivity is implied.
 */
typedef struct XRSLAMDenseSemanticPoint {
    uint64_t session_id;
    int32_t voxel_ix, voxel_iy, voxel_iz;
    double xyz[3];
    uint8_t rgb[3];
    uint8_t semantic_class;
    double consensus;
    uint32_t observation_count;
    uint8_t stable;
} XRSLAMDenseSemanticPoint;

size_t XRSLAMGetDenseSemanticMapSnapshot(XRSLAMDenseSemanticPoint *dst,
                                         size_t capacity, size_t *total,
                                         int stable_only);

// Session-local robot-body observation poses; map_aligned is always false
// until verified map-frame session alignment is implemented. Call with
// dst=NULL/capacity=0 to discover total (limited by max_results).
typedef struct XRSLAMSemanticKeyframeHit {
    uint64_t session_id, frame_id;
    double timestamp;
    uint32_t class_id;
    double coverage, mean_confidence;
    double body_position[3];
    double body_quaternion_xyzw[4];
    int map_aligned;
    char taxonomy[64];
    char model_id[128];
} XRSLAMSemanticKeyframeHit;

size_t XRSLAMQuerySemanticKeyframes(
    const char *class_name, double min_mean_confidence,
    double min_pixel_coverage, size_t max_results,
    XRSLAMSemanticKeyframeHit *dst, size_t capacity, size_t *total);

/* Separate from GLOBAL_BODY_POSE and its local-pose fallback.
 * This exposes T_map_vio, NOT T_map_odom. A return of 1 means an XRSLAM
 * instance exists; the valid field is the ONLY localization authority.
 * valid=0 must never be interpreted as identity. Do not call concurrently
 * with XRSLAMDestroy. map_id is stable identity, not a TF frame name.
 */
typedef struct XRSLAMVerifiedMapAlignment {
    int valid;
    int reference_map_loaded;
    uint64_t tracking_session_epoch;
    uint64_t reference_session_id;
    double verified_timestamp;
    double yaw_rad;
    double translation[3];
    char map_id[128];
} XRSLAMVerifiedMapAlignment;

int XRSLAMGetVerifiedMapAlignment(XRSLAMVerifiedMapAlignment *out);

/**
 * @brief destroy SLAM system
 */
void XRSLAMDestroy();

#ifdef __cplusplus
}
#endif

#endif // _XRSLAM_H_
