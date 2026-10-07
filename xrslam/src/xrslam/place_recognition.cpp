#include <xrslam/place_recognition.h>

namespace xrslam {

bool PlaceKeyframeStore::add(const PlaceKeyframe &keyframe) {
    if (key_to_index_.count(keyframe.key) != 0)
        return false;

    const size_t index = keyframes_.size();
    keyframes_.push_back(keyframe);
    key_to_index_.emplace(keyframe.key, index);
    return true;
}

const PlaceKeyframe *PlaceKeyframeStore::find(PlaceKey key) const {
    const auto it = key_to_index_.find(key);
    if (it == key_to_index_.end())
        return nullptr;
    return &keyframes_[it->second];
}

void PlaceKeyframeStore::clear() {
    keyframes_.clear();
    key_to_index_.clear();
}

} // namespace xrslam
