#ifndef XRSLAM_TRACK_H
#define XRSLAM_TRACK_H

#include <xrslam/common.h>
#include <xrslam/estimation/state.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>
#include <xrslam/utility/identifiable.h>
#include <xrslam/utility/tag.h>

namespace xrslam {

enum TrackTag : int {
    TT_VALID = 0,
    TT_TRIANGULATED,
    TT_FIX_INVD,
    TT_TRASH,
    TT_STATIC,
    TT_OUTLIER,
    TT_TEMP
};

class Track : public Tagged<TrackTag>, public Identifiable<Track> {
    friend class Map;
    size_t map_index;
    Map *map;
    Track();

  public:
    Track(const Map::construct_by_map_t &) : Track() {}

    virtual ~Track();

    size_t keypoint_num() const { return keypoint_refs.size(); }

    std::pair<Frame *, size_t> first_keypoint() const {
        return *keypoint_refs.begin();
    }

    std::pair<Frame *, size_t> last_keypoint() const {
        return *keypoint_refs.rbegin();
    }

    Frame *first_frame() const { return keypoint_refs.begin()->first; }

    Frame *last_frame() const { return keypoint_refs.rbegin()->first; }

    const std::map<Frame *, size_t, compare<Frame *>> &keypoint_map() const {
        return keypoint_refs;
    }

    bool has_keypoint(Frame *frame) const {
        return keypoint_refs.count(frame) > 0;
    }

    size_t get_keypoint_index(Frame *frame) const {
        if (has_keypoint(frame)) {
            return keypoint_refs.at(frame);
        } else {
            return nil();
        }
    }

    const vector<3> &get_keypoint(Frame *frame) const;
    void add_keypoint(Frame *frame, size_t keypoint_index);
    void remove_keypoint(Frame *frame, bool suicide_if_empty = true);

    std::optional<vector<3>> triangulate();
    double triangulation_angle(const vector<3> &p) const;

    vector<3> get_landmark_point() const;
    void set_landmark_point(const vector<3> &p);

    std::unique_lock<std::mutex> lock() const { return map->lock(); }

    LandmarkState landmark;
    bool has_depth_prior = false;
    double depth_prior_inv_depth = 0.0;
    double depth_prior_sqrt_info = 0.0;
    DepthSource depth_prior_source = DepthSource::SENSOR_METRIC;
    double depth_prior_confidence = 1.0;

    // Recovery-only exact world-space landmark copied from the optimized
    // sliding-window map. This must not be folded into landmark.inv_depth:
    // inverse depth is anchored to this Track's first feature-map observation,
    // whose pose can differ from the optimized sliding-window anchor pose.
    std::optional<vector<3>> recovery_landmark_world;

    size_t m_life = 0;

  private:
    std::map<Frame *, size_t, compare<Frame *>> keypoint_refs;
};

} // namespace xrslam

#endif // XRSLAM_TRACK_H
