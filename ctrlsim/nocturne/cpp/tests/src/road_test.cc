// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "road.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace {

TEST(RoadLineTest, SampleRoadPointTest) {
  constexpr RoadType kRoadType = RoadType::kLane;
  constexpr int64_t num_geometry_points = 10;
  std::vector<geometry::Vector2D> geometry_points;
  geometry_points.reserve(num_geometry_points);
  for (int64_t i = 0; i < num_geometry_points; ++i) {
    geometry_points.emplace_back(static_cast<float>(i), static_cast<float>(i));
  }

  for (int64_t i = 1; i < num_geometry_points; ++i) {
    const int64_t num_sampled_points = (num_geometry_points + i - 2) / i + 1;
    RoadLine road_line(kRoadType, geometry_points, /*sample_every_n=*/i);
    const std::vector<RoadPoint>& road_points = road_line.road_points();
    ASSERT_EQ(road_points.size(), num_sampled_points);
    for (int64_t j = 0; j < num_sampled_points - 1; ++j) {
      const int64_t p = i * j;
      const int64_t q = std::min(i * (j + 1), num_geometry_points - 1);
      EXPECT_FLOAT_EQ(road_points[j].position().x(), geometry_points[p].x());
      EXPECT_FLOAT_EQ(road_points[j].position().y(), geometry_points[p].y());
      EXPECT_FLOAT_EQ(road_points[j].neighbor_position().x(),
                      geometry_points[q].x());
      EXPECT_FLOAT_EQ(road_points[j].neighbor_position().y(),
                      geometry_points[q].y());
    }
    EXPECT_FLOAT_EQ(road_points.back().position().x(),
                    geometry_points.back().x());
    EXPECT_FLOAT_EQ(road_points.back().position().y(),
                    geometry_points.back().y());
    EXPECT_FLOAT_EQ(road_points.back().neighbor_position().x(),
                    geometry_points.back().x());
    EXPECT_FLOAT_EQ(road_points.back().neighbor_position().y(),
                    geometry_points.back().y());
  }
}

TEST(RoadLineTest, EmptyGeometryProducesNoRoadPoints) {
  const RoadLine road_line(RoadType::kLane, std::vector<geometry::Vector2D>{});

  EXPECT_TRUE(road_line.road_points().empty());
  EXPECT_TRUE(road_line.geometry_points().empty());
}

TEST(RoadLineTest, SingleGeometryPointProducesSelfNeighborRoadPoint) {
  const geometry::Vector2D only_point(3.0f, 4.0f);
  const RoadLine road_line(RoadType::kLane, std::vector<geometry::Vector2D>{only_point});

  ASSERT_EQ(road_line.road_points().size(), 1U);
  EXPECT_FLOAT_EQ(road_line.road_points()[0].position().x(), only_point.x());
  EXPECT_FLOAT_EQ(road_line.road_points()[0].position().y(), only_point.y());
  EXPECT_FLOAT_EQ(road_line.road_points()[0].neighbor_position().x(),
                  only_point.x());
  EXPECT_FLOAT_EQ(road_line.road_points()[0].neighbor_position().y(),
                  only_point.y());
}

}  // namespace
}  // namespace nocturne
