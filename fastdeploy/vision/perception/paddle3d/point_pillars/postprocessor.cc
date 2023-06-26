// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/vision/perception/paddle3d/point_pillars/postprocessor.h"

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace perception {

PointPillarPostprocessor::PointPillarsPostprocessor() {}

bool PointPillarPostprocessor::Run(const std::vector<FDTensor>& tensors,
                             std::vector<PerceptionResult>* results) {
  // TODO: PointPillar Postprocessor need to be implemented.
  return true;
}

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
