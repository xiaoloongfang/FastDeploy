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

#include "fastdeploy/vision/perception/paddle3d/point_pillars/preprocessor.h"

#include "fastdeploy/function/concat.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace perception {

PointPillarsPreprocessor::PointPillarsPreprocessor(const std::string& config_file) {
  config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create Paddle3DDetPreprocessor.");
  initialized_ = true;
}

bool PointPillarsPreprocessor::BuildPreprocessPipelineFromConfig() {
  // TODO: PointPillar Preprocessor need to be implemented.
  return true;
}

// TODO : The definition of Apply function will be modified to be  
// Apply(FDTensor* voxels_batch, FDTensor* coords_batch, 
// FDTensor* num_points_per_voxel_batch, std::vector<FDTensor>* outputs)
bool PointPillarsPreprocessor::Apply(FDTensor* voxels_batch, 
                              FDTensor* coords_batch, 
                              FDTensor* num_points_per_voxel_batch,
                              std::vector<FDTensor>* outputs) {
  // TODO: PointPillar Preprocessor need to be implemented.
  return true;
}

// function Run will invoke Apply function.
bool ProcessorManager::Run(const std::vector<FDTensor>& voxels_batch,
                          const std::vector<FDTensor>& coords_batch,
                          const std::vector<FDTensor>& num_points_per_voxel_batch,
                          std::vector<FDTensor>* outputs) {
  //TODO : PreApply needs to be implemented
  PreApply(&image_batch);
  bool ret = Apply(&voxels_batch, 
                  &coords_batch, 
                  &num_points_per_voxel_batch, 
                  outputs);
  PostApply();
  return ret;
}

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
