#include <xrslam/extra/usearch_place_database.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

xrslam::PlaceDescriptor make_descriptor(float x, float y) {
    xrslam::PlaceDescriptor descriptor;
    descriptor.values.assign(
        xrslam::extra::USearchPlaceDatabase::kDescriptorDimension, 0.0f);
    descriptor.values[0] = x;
    descriptor.values[1] = y;
    return descriptor;
}

bool near(float actual, float expected, float tolerance = 1.0e-4f) {
    return std::abs(actual - expected) <= tolerance;
}

int fail(const char *message) {
    std::fprintf(stderr, "USearch PlaceDatabase smoke test: FAIL: %s\n", message);
    return EXIT_FAILURE;
}

template <typename T, typename = void>
struct has_frame_id : std::false_type {};

template <typename T>
struct has_frame_id<T, std::void_t<decltype(std::declval<const T &>().frame_id)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_keyframe_id : std::false_type {};

template <typename T>
struct has_keyframe_id<
    T, std::void_t<decltype(std::declval<const T &>().keyframe_id)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_id : std::false_type {};

template <typename T>
struct has_id<T, std::void_t<decltype(std::declval<const T &>().id)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_key : std::false_type {};

template <typename T>
struct has_key<T, std::void_t<decltype(std::declval<const T &>().key)>>
    : std::true_type {};

template <typename T>
size_t candidate_id(const T &candidate) {
    if constexpr (has_frame_id<T>::value) {
        return static_cast<size_t>(candidate.frame_id);
    } else if constexpr (has_keyframe_id<T>::value) {
        return static_cast<size_t>(candidate.keyframe_id);
    } else if constexpr (has_id<T>::value) {
        return static_cast<size_t>(candidate.id);
    } else if constexpr (has_key<T>::value) {
        return static_cast<size_t>(candidate.key);
    } else {
        static_assert(has_frame_id<T>::value || has_keyframe_id<T>::value ||
                          has_id<T>::value || has_key<T>::value,
                      "PlaceCandidate key member is not recognized");
        return 0;
    }
}

} // namespace

int main() {
    xrslam::extra::USearchPlaceDatabase database;

    if (database.dimension() !=
        xrslam::extra::USearchPlaceDatabase::kDescriptorDimension)
        return fail("database reports the wrong descriptor dimension");
    if (database.size() != 0)
        return fail("new database is not empty");

    const auto empty_query = make_descriptor(1.0f, 0.0f);
    if (!database.search(empty_query, 5).empty())
        return fail("empty database returned candidates");

    database.reserve(8);

    const auto a = make_descriptor(1.0f, 0.0f);
    const auto b = make_descriptor(0.9f, std::sqrt(0.19f));
    const auto c = make_descriptor(0.0f, 1.0f);

    if (!database.add(100, a))
        return fail("failed to add frame 100");
    if (!database.add(200, b))
        return fail("failed to add frame 200");
    if (!database.add(300, c))
        return fail("failed to add frame 300");
    if (database.size() != 3)
        return fail("size after insertion is not 3");

    if (database.add(100, a))
        return fail("duplicate frame ID was accepted");
    if (database.size() != 3)
        return fail("duplicate insertion changed database size");

    auto top1 = database.search(a, 1);
    if (top1.size() != 1 || candidate_id(top1[0]) != 100 ||
        !near(top1[0].distance, 0.0f))
        return fail("top-1 query is incorrect");

    auto top2 = database.search(a, 2);
    if (top2.size() != 2 || candidate_id(top2[0]) != 100 ||
        candidate_id(top2[1]) != 200)
        return fail("top-2 key ordering is incorrect");
    if (!near(top2[0].distance, 0.0f) ||
        !near(top2[1].distance, 0.2f))
        return fail("top-2 squared-L2 distances are incorrect");

    auto all = database.search(a, 10);
    if (all.size() != 3 || candidate_id(all[0]) != 100 ||
        candidate_id(all[1]) != 200 || candidate_id(all[2]) != 300)
        return fail("k > size result ordering is incorrect");
    if (!near(all[2].distance, 2.0f))
        return fail("third squared-L2 distance is incorrect");

    if (!database.search(a, 0).empty())
        return fail("k=0 returned candidates");

    xrslam::PlaceDescriptor wrong_dimension;
    wrong_dimension.values.assign(
        xrslam::extra::USearchPlaceDatabase::kDescriptorDimension - 1,
        0.0f);
    if (database.add(400, wrong_dimension))
        return fail("invalid descriptor dimension was accepted");
    if (!database.search(wrong_dimension, 5).empty())
        return fail("invalid query descriptor returned candidates");

    auto factory_database = xrslam::extra::make_usearch_place_database();
    if (!factory_database ||
        factory_database->dimension() !=
            xrslam::extra::USearchPlaceDatabase::kDescriptorDimension ||
        factory_database->size() != 0)
        return fail("PlaceDatabase factory returned an invalid database");

    std::printf("dimension: %zu\n",
                xrslam::extra::USearchPlaceDatabase::kDescriptorDimension);
    std::printf("index size: %zu\n", database.size());
    for (size_t i = 0; i < all.size(); ++i) {
        std::printf("rank=%zu frame_id=%zu distance=%.8f\n",
                    i, candidate_id(all[i]), all[i].distance);
    }

    database.clear();
    if (database.size() != 0 || !database.search(a, 5).empty())
        return fail("clear did not empty the database");

    std::printf("USearch PlaceDatabase smoke test: PASS\n");
    return EXIT_SUCCESS;
}
