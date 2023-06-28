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

#include "fastdeploy/vision/perception/paddle3d/point_pillars/point_pillars.h"

namespace fastdeploy {
namespace vision {
namespace perception {

PointPillars::PointPillars(const std::string& model_file, const std::string& params_file,
            const RuntimeOption& custom_option, const ModelFormat& model_format){
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PointPillars::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

// Todo : The definition of Predict should be modified to 
// Predict(const FDTensor& voxels, const FDTensor& coords, const FDTensor& num_points_per_voxel, PerceptionResult* result)
// bool PointPillars::Predict(const cv::Mat& im, PerceptionResult* result) {
bool PointPillars::Predict(FDTensor& voxels, 
                          FDTensor& coords, 
                          FDTensor& num_points_per_voxel, 
                          PerceptionResult* result){
  std::vector<PerceptionResult> results;
  voxels.ExpandDim();
  coords.ExpandDim();
  num_points_per_voxel.ExpandDim();
  if (!BatchPredict(voxels, coords, num_points_per_voxel, &results)) {
    return false;
  }
  if (results.size()) {
    *result = std::move(results[0]);
  }
  return true;
}

// Todo : The definition of BatchPredict should modified according to Predict function.
// bool PointPillars::BatchPredict(const std::vector<cv::Mat>& images,
//                          std::vector<PerceptionResult>* results) {
bool PointPillars::BatchPredict(FDTensor& voxels_batch, FDTensor& coords_batch,
                              FDTensor& num_points_per_voxel_batch, 
                              std::vector<PerceptionResult>* results) {

  if (!preprocessor_.Run(voxels_batch, 
                        coords_batch, 
                        num_points_per_voxel_batch, 
                        &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input tensor." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  reused_input_tensors_[1].name = InputInfoOfRuntime(1).name;
  reused_input_tensors_[2].name = InputInfoOfRuntime(2).name;

  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
