#include <npy_depth.h>

#include <cstdint>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace {

template <typename T> bool read_scalar(std::ifstream &input, T &value) {
    input.read(reinterpret_cast<char *>(&value), sizeof(T));
    return input.good();
}

} // namespace

bool load_npy_float32_2d(const std::string &filename, cv::Mat &output) {
    output.release();

    std::ifstream input(filename, std::ios::binary);
    if (!input)
        return false;

    unsigned char magic[6] = {};
    input.read(reinterpret_cast<char *>(magic), sizeof(magic));
    const unsigned char expected[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    for (size_t i = 0; i < sizeof(magic); ++i) {
        if (magic[i] != expected[i])
            return false;
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    if (!read_scalar(input, major) || !read_scalar(input, minor))
        return false;
    (void)minor;

    uint32_t header_length = 0;
    if (major == 1) {
        uint16_t length16 = 0;
        if (!read_scalar(input, length16))
            return false;
        header_length = length16;
    } else if (major == 2 || major == 3) {
        if (!read_scalar(input, header_length))
            return false;
    } else {
        return false;
    }

    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input.good())
        return false;

    if (header.find("'fortran_order': True") != std::string::npos ||
        header.find("\"fortran_order\": True") != std::string::npos)
        return false;

    std::smatch match;
    const std::regex descr_re(
        R"(['"]descr['"]\s*:\s*['"]([^'"]+)['"])");
    if (!std::regex_search(header, match, descr_re))
        return false;
    const std::string descr = match[1].str();
    if (descr != "<f4" && descr != "=f4" && descr != "|f4")
        return false;

    const std::regex shape_re(
        R"(['"]shape['"]\s*:\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*\))");
    if (!std::regex_search(header, match, shape_re))
        return false;

    const int rows = std::stoi(match[1].str());
    const int cols = std::stoi(match[2].str());
    if (rows <= 0 || cols <= 0)
        return false;

    output.create(rows, cols, CV_32FC1);
    const std::streamsize bytes =
        static_cast<std::streamsize>(rows) * cols * sizeof(float);
    input.read(reinterpret_cast<char *>(output.data), bytes);
    if (input.gcount() != bytes) {
        output.release();
        return false;
    }

    return true;
}
