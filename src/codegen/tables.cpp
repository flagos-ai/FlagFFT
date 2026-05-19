#include "flagfft/core.hpp"

namespace flagfft {

std::pair<std::vector<int64_t>, std::vector<int64_t>> decode_stage_codelet(
    int64_t codelet, const std::vector<int64_t> &radices, int64_t stage) {
    std::vector<int64_t> prev_freq;
    int64_t rem = codelet;
    for (int64_t axis = 0; axis < stage; ++axis) {
        prev_freq.push_back(rem % radices[static_cast<std::size_t>(axis)]);
        rem /= radices[static_cast<std::size_t>(axis)];
    }

    std::vector<int64_t> future_time(radices.size(), 0);
    for (int64_t axis = static_cast<int64_t>(radices.size()) - 1; axis > stage; --axis) {
        future_time[static_cast<std::size_t>(axis)] = rem % radices[static_cast<std::size_t>(axis)];
        rem /= radices[static_cast<std::size_t>(axis)];
    }
    if (rem != 0) {
        throw std::runtime_error("stage codelet decode overflow");
    }
    return {prev_freq, future_time};
}

int64_t mixed_radix_value(const std::vector<int64_t> &digits,
                          const std::vector<int64_t> &radices,
                          std::size_t limit) {
    int64_t value = 0;
    int64_t stride = 1;
    for (std::size_t axis = 0; axis < limit; ++axis) {
        value += digits[axis] * stride;
        stride *= radices[axis];
    }
    return value;
}

std::pair<std::vector<float>, std::vector<float>> build_stage_twiddles(
    const std::vector<int64_t> &radices,
    int64_t stage,
    int64_t lanes,
    const std::string &direction) {
    int64_t n = product(radices);
    int64_t elems_per_lane = n / lanes;
    int64_t radix = radices[static_cast<std::size_t>(stage)];
    int64_t denom = 1;
    for (int64_t axis = 0; axis <= stage; ++axis) {
        denom *= radices[static_cast<std::size_t>(axis)];
    }

    std::vector<float> tw_r(static_cast<std::size_t>(n));
    std::vector<float> tw_i(static_cast<std::size_t>(n));
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t elem = 0; elem < elems_per_lane; ++elem) {
            int64_t group = elem / radix;
            int64_t digit = elem % radix;
            int64_t codelet = lane + lanes * group;
            auto decoded = decode_stage_codelet(codelet, radices, stage);
            int64_t prefix = mixed_radix_value(decoded.first, radices, decoded.first.size());
            double sign = direction == "inverse" ? 1.0 : -1.0;
            double angle = sign * 2.0 * kPi * static_cast<double>(prefix * digit) /
                           static_cast<double>(denom);
            std::size_t index = static_cast<std::size_t>(lane + lanes * elem);
            tw_r[index] = static_cast<float>(std::cos(angle));
            tw_i[index] = static_cast<float>(std::sin(angle));
        }
    }
    return {tw_r, tw_i};
}

std::pair<std::vector<float>, std::vector<float>> build_dft_matrix(int64_t radix,
                                                                   const std::string &direction) {
    std::vector<float> dft_r(static_cast<std::size_t>(radix * radix));
    std::vector<float> dft_i(static_cast<std::size_t>(radix * radix));
    double sign = direction == "inverse" ? 1.0 : -1.0;
    for (int64_t k = 0; k < radix; ++k) {
        for (int64_t n = 0; n < radix; ++n) {
            double angle = sign * 2.0 * kPi * static_cast<double>(k * n) /
                           static_cast<double>(radix);
            std::size_t index = static_cast<std::size_t>(k * radix + n);
            dft_r[index] = static_cast<float>(std::cos(angle));
            dft_i[index] = static_cast<float>(std::sin(angle));
        }
    }
    return {dft_r, dft_i};
}

DeviceAllocation allocate_device_bytes(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    CUdeviceptr ptr = 0;
    cuda_check(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
    return DeviceAllocation(ptr, bytes);
}

DeviceAllocation device_allocation_from_floats(const std::vector<float> &values) {
    DeviceAllocation allocation = allocate_device_bytes(values.size() * sizeof(float));
    if (allocation.ptr != 0) {
        cuda_check(cuMemcpyHtoD(allocation.ptr, values.data(), allocation.bytes), "cuMemcpyHtoD");
    }
    return allocation;
}

DeviceAllocation build_raw_four_step_twiddle(const FFTRequest &request, int64_t n1, int64_t n2) {
    std::vector<float> interleaved(static_cast<std::size_t>(n1 * n2 * 2));
    for (int64_t row = 0; row < n2; ++row) {
        for (int64_t col = 0; col < n1; ++col) {
            double angle = -2.0 * kPi * static_cast<double>(row * col) /
                           static_cast<double>(n1 * n2);
            if (request.direction == "inverse") {
                angle = -angle;
            }
            std::size_t index = static_cast<std::size_t>((row * n1 + col) * 2);
            interleaved[index] = static_cast<float>(std::cos(angle));
            interleaved[index + 1] = static_cast<float>(std::sin(angle));
        }
    }
    return device_allocation_from_floats(interleaved);
}

DeviceAllocation build_raw_bluestein_chirp(const FFTRequest &request, int64_t n, bool inverse_sign) {
    (void)request;
    std::vector<float> interleaved(static_cast<std::size_t>(n * 2));
    double sign = inverse_sign ? 1.0 : -1.0;
    for (int64_t idx = 0; idx < n; ++idx) {
        double reduced = std::fmod(static_cast<double>(idx) * static_cast<double>(idx),
                                   static_cast<double>(2 * n));
        double angle = sign * kPi * reduced / static_cast<double>(n);
        std::size_t offset = static_cast<std::size_t>(idx * 2);
        interleaved[offset] = static_cast<float>(std::cos(angle));
        interleaved[offset + 1] = static_cast<float>(std::sin(angle));
    }
    return device_allocation_from_floats(interleaved);
}

DeviceAllocation build_raw_bluestein_b(const FFTRequest &request, int64_t n, int64_t m) {
    std::vector<float> interleaved(static_cast<std::size_t>(m * 2), 0.0f);
    double sign = request.direction == "inverse" ? -1.0 : 1.0;
    for (int64_t idx = 0; idx < n; ++idx) {
        double reduced = std::fmod(static_cast<double>(idx) * static_cast<double>(idx),
                                   static_cast<double>(2 * n));
        double angle = sign * kPi * reduced / static_cast<double>(n);
        float r = static_cast<float>(std::cos(angle));
        float i = static_cast<float>(std::sin(angle));
        std::size_t offset = static_cast<std::size_t>(idx * 2);
        interleaved[offset] = r;
        interleaved[offset + 1] = i;
        if (idx != 0) {
            std::size_t mirror = static_cast<std::size_t>((m - idx) * 2);
            interleaved[mirror] = r;
            interleaved[mirror + 1] = i;
        }
    }
    return device_allocation_from_floats(interleaved);
}

std::vector<DeviceAllocation> build_raw_leaf_tables(const LeafPlanNode &leaf,
                                                    const FFTRequest &request) {
    std::vector<DeviceAllocation> tables;
    for (std::size_t stage = 1; stage < leaf.factors.size(); ++stage) {
        auto twiddles = build_stage_twiddles(
            leaf.factors, static_cast<int64_t>(stage), leaf.lanes, request.direction);
        tables.push_back(device_allocation_from_floats(twiddles.first));
        tables.push_back(device_allocation_from_floats(twiddles.second));
    }
    for (int64_t radix : leaf.generic_radices) {
        auto dft = build_dft_matrix(radix, request.direction);
        tables.push_back(device_allocation_from_floats(dft.first));
        tables.push_back(device_allocation_from_floats(dft.second));
    }
    return tables;
}

}  // namespace flagfft
