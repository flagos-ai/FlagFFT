#include "runtime/check.hpp"

#include <sstream>
#include <stdexcept>

namespace flagfft::runtime {

void check(CUresult result, const std::string &context) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream out;
    out << context << " failed";
    if (name != nullptr) {
        out << " (" << name << ")";
    }
    if (message != nullptr) {
        out << ": " << message;
    }
    throw std::runtime_error(out.str());
}

}  // namespace flagfft::runtime
