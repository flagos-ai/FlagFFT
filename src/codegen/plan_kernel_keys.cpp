#include "flagfft/core.hpp"

namespace flagfft {

FFTRequest forward_child_request(const FFTRequest &request) {
    FFTRequest child = request;
    child.direction = "forward";
    child.norm = "backward";
    return child;
}

std::string triton_target_for_request(const FFTRequest &request) {
    std::string arch = request.device_arch;
    if (arch.rfind("sm_", 0) == 0) {
        arch.erase(0, 3);
    }
    return "cuda:" + arch + ":32";
}

}  // namespace flagfft
