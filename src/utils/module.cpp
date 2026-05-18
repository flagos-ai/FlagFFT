#include "flagfft/core.hpp"

NB_MODULE(_flagfft_core, m) {
    m.doc() = "C++ request, plan, and execution core for FlagFFT";

    m.def("fft", &flagfft::fft, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "Execute a 1-D FFT through the C++ plan cache");
    m.def("ifft", &flagfft::ifft, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "Execute a 1-D inverse FFT through the C++ plan cache");
    m.def("fft_with_plan", &flagfft::fft_with_plan, "input"_a, "plan"_a,
          "n"_a = nb::none(), "dim"_a = -1, "norm"_a = nb::none(),
          "Execute a 1-D FFT with an explicit tune plan");
    m.def("ifft_with_plan", &flagfft::ifft_with_plan, "input"_a, "plan"_a,
          "n"_a = nb::none(), "dim"_a = -1, "norm"_a = nb::none(),
          "Execute a 1-D inverse FFT with an explicit tune plan");
    m.def("debug_request", &flagfft::debug_request, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "direction"_a = "forward");
    m.def("debug_keys", &flagfft::debug_keys, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "direction"_a = "forward");
    m.def("debug_plan", &flagfft::debug_plan, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "direction"_a = "forward");
    m.def("debug_resolved_plan", &flagfft::debug_resolved_plan, "input"_a,
          "n"_a = nb::none(), "dim"_a = -1, "norm"_a = nb::none(), "direction"_a = "forward");
    m.def("debug_forced_plan", &flagfft::debug_forced_plan, "input"_a, "plan"_a,
          "n"_a = nb::none(), "dim"_a = -1, "norm"_a = nb::none(), "direction"_a = "forward");
    m.def("enumerate_plan_candidates", &flagfft::enumerate_plan_candidates, "input"_a,
          "n"_a = nb::none(), "dim"_a = -1, "norm"_a = nb::none(), "direction"_a = "forward");
    m.def("tune_fingerprints", &flagfft::tune_fingerprints);
    m.def("clear_plan_cache", []() { flagfft::plan_cache().clear(); });
    m.def("cache_info", []() { return flagfft::plan_cache().info(); });
    m.def("cache_keys", []() { return flagfft::plan_cache().keys(); });
}
