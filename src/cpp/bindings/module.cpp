#include "flagfft/core.hpp"

NB_MODULE(_flagfft_core, m) {
    m.doc() = "C++ request, plan, and execution core for FlagFFT";

    m.def("fft", &flagfft::fft, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "Execute a 1-D FFT through the C++ plan cache");
    m.def("debug_request", &flagfft::debug_request, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none());
    m.def("debug_plan_key", &flagfft::debug_plan_key, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none());
    m.def("debug_plan", &flagfft::debug_plan, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none());
    m.def("clear_plan_cache", []() { flagfft::plan_cache().clear(); });
    m.def("cache_info", []() { return flagfft::plan_cache().info(); });
    m.def("cache_keys", []() { return flagfft::plan_cache().keys(); });
}
