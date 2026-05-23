#pragma once

#include <cstddef>
#include <vector>

#include "types.hpp"

namespace flagfft::runtime {

class Memory {
public:
    Memory() = default;
    explicit Memory(std::size_t bytes);
    ~Memory();

    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;
    Memory(Memory &&other) noexcept;
    Memory &operator=(Memory &&other) noexcept;

    void reset();
    DevicePtr get() const noexcept;
    std::size_t size() const noexcept;

    static Memory allocate(std::size_t bytes);
    static Memory from_floats(const std::vector<float> &values);
    static Memory from_doubles(const std::vector<double> &values);

private:
    DevicePtr ptr_ = 0;
    std::size_t bytes_ = 0;
};

}  // namespace flagfft::runtime
