#include "flagfft/core.hpp"

namespace flagfft {

FFTRequest forward_child_request(const FFTRequest &request) {
  FFTRequest child = request;
  child.direction = "forward";
  child.norm = "backward";
  return child;
}

std::string triton_target_for_request(const FFTRequest &request) {
  return adaptor::triton_target(request.device_arch);
}

}  // namespace flagfft
