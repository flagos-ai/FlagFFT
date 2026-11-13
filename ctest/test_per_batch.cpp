// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "flagfft/core.hpp"
#include "flagfft_test.h"
#include <vector>

namespace {
struct RecordedCall {
  flagfft::adaptor::DevicePtr input, output;
  int64_t batch, input_distance, output_distance;
};
struct RecordingNode final : flagfft::CompiledRawNode {
  mutable std::vector<RecordedCall> calls;
  int fail_call = -1;
  flagfftResult execute(flagfft::adaptor::DevicePtr input, flagfft::adaptor::DevicePtr output,
                        const flagfft::RawExecutionContext &context) const override {
    calls.push_back({input,output,context.batch,context.input_distance,context.output_distance});
    return static_cast<int>(calls.size()) == fail_call ? FLAGFFT_EXEC_FAILED : FLAGFFT_SUCCESS;
  }
  std::string describe() const override { return "RecordingNode"; }
};
}

TEST(PerBatchNode, SeparatesRealInputAndComplexOutputStrides) {
  auto child=std::make_shared<RecordingNode>();
  flagfft::CompiledRawPerBatchNode node(child,16,9,4,8);
  flagfft::FFTRequest request;
  flagfft::RawExecutionContext context{request,nullptr,2};
  ASSERT_EQ(node.execute(10000,20000,context),FLAGFFT_SUCCESS);
  ASSERT_EQ(child->calls.size(),2);
  EXPECT_EQ(child->calls[1].input,10064);
  EXPECT_EQ(child->calls[1].output,20072);
  EXPECT_EQ(child->calls[1].batch,1);
  EXPECT_EQ(child->calls[1].input_distance,0);
  EXPECT_EQ(child->calls[1].output_distance,0);
  child->calls.clear();
  context.input_distance=20;
  context.output_distance=12;
  ASSERT_EQ(node.execute(10000,20000,context),FLAGFFT_SUCCESS);
  EXPECT_EQ(child->calls[1].input,10080);
  EXPECT_EQ(child->calls[1].output,20096);
}

TEST(PerBatchNode, PropagatesFailureBeforeStartingTheNextTransform) {
  auto child=std::make_shared<RecordingNode>();
  child->fail_call=2;
  flagfft::CompiledRawPerBatchNode node(child,16,16,8,8);
  flagfft::FFTRequest request;
  flagfft::RawExecutionContext context{request,nullptr,4};
  EXPECT_EQ(node.execute(10000,20000,context),FLAGFFT_EXEC_FAILED);
  EXPECT_EQ(child->calls.size(),2);
}
