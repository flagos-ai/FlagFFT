#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "flagfft/core.hpp"

namespace {

flagfft::FFTRequest make_request(int64_t n, int64_t batch = 1, std::string direction = "forward") {
    flagfft::FFTRequest request;
    request.fft_length = n;
    request.input_shape = {batch, n};
    request.input_strides = {n, 1};
    request.requested_n = n;
    request.raw_dim = 1;
    request.normalized_dim = 1;
    request.norm = "backward";
    request.input_dtype = "complex64";
    request.output_dtype = "complex64";
    request.device_type = "cpu";
    request.device_index = 0;
    request.device_arch = "test";
    request.input_layout = "contiguous";
    request.requires_contiguous_copy = false;
    request.direction = std::move(direction);
    request.batch = batch;
    return request;
}

}  // namespace

TEST(FlagFFTPlan, SmallFactorableLengthUsesLeafPlan) {
    flagfft::PlanBuilder builder;
    auto request = make_request(16, 4);
    auto root = builder.build(request.requested_n, request);

    ASSERT_EQ(root->kind, flagfft::PlanNodeKind::CtLeaf);
    auto leaf = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(root);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->length, 16);
    EXPECT_EQ(leaf->remainder, 1);
    EXPECT_TRUE(leaf->generic_radices.empty());
    EXPECT_GT(leaf->lanes, 0);
    EXPECT_GT(leaf->num_warps, 0);
}

TEST(FlagFFTPlan, CompositeLengthBeyondLeafLimitUsesFourStepPlan) {
    flagfft::PlanBuilder builder;
    auto request = make_request(8192, 2);
    auto root = builder.build(request.requested_n, request);

    ASSERT_EQ(root->kind, flagfft::PlanNodeKind::FourStep);
    auto four_step = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(root);
    ASSERT_NE(four_step, nullptr);
    EXPECT_EQ(four_step->length, 8192);
    EXPECT_EQ(four_step->n1 * four_step->n2, 8192);
    EXPECT_NE(four_step->row_plan, nullptr);
    EXPECT_NE(four_step->col_plan, nullptr);
}

TEST(FlagFFTPlan, UnsupportedPrimeLengthFallsBackToBluesteinPlan) {
    flagfft::PlanBuilder builder;
    auto request = make_request(331, 1);
    auto root = builder.build(request.requested_n, request);

    ASSERT_EQ(root->kind, flagfft::PlanNodeKind::Bluestein);
    auto bluestein = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(root);
    ASSERT_NE(bluestein, nullptr);
    EXPECT_EQ(bluestein->length, 331);
    EXPECT_GE(bluestein->conv_length, 2 * 331 - 1);
    EXPECT_NE(bluestein->fft_plan, nullptr);
}

TEST(FlagFFTPlan, PlanKeyCapturesRouteStructure) {
    flagfft::PlanBuilder builder;
    auto request = make_request(8192, 1);
    auto root = builder.build(request.requested_n, request);
    flagfft::PlanKey key = flagfft::PlanKey::from_node(root);

    EXPECT_EQ(key.schema_version, flagfft::kPlanSchemaVersion);
    EXPECT_EQ(key.root_kind, flagfft::PlanNodeKind::FourStep);
    EXPECT_EQ(key.length, 8192);
    EXPECT_EQ(key.n1 * key.n2, 8192);
    EXPECT_EQ(key.child_keys.size(), 2);
    EXPECT_NE(key.repr().find("kind=four_step"), std::string::npos);
}

TEST(FlagFFTPlan, KernelKeysSeparateForwardAndInverseArtifacts) {
    flagfft::KernelKey forward = flagfft::KernelKey::leaf(
        "cuda:80:32", "forward", 16, {16}, 1, 1, {}, 0);
    flagfft::KernelKey inverse = flagfft::KernelKey::leaf(
        "cuda:80:32", "inverse", 16, {16}, 1, 1, {}, 0);

    EXPECT_FALSE(forward == inverse);
    EXPECT_NE(flagfft::KernelKeyHash{}(forward), flagfft::KernelKeyHash{}(inverse));
    EXPECT_NE(forward.repr(), inverse.repr());
}
