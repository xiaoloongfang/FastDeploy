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

#include "fastdeploy/pybind/main.h"

namespace fastdeploy {
void BindPointPillars(pybind11::module& m) {
  pybind11::class_<vision::perception::PointPillarsPreprocessor,
                   vision::ProcessorManager>(m, "PointPillarsPreprocessor")
      .def(pybind11::init<>())
      .def("run", [](vision::perception::PointPillarsPreprocessor& self,
                     FDTensor& voxels, FDTensor& coords, FDTensor& num_points_per_voxel) {
        std::vector<FDTensor> outputs;
        if (!self.Run(voxels, coords, num_points_per_voxel, &outputs)) {
          throw std::runtime_error(
              "Failed to preprocess the input data in PointPillarsPreprocessor.");
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
          outputs[i].StopSharing();
        }
        return outputs;
      });

  pybind11::class_<vision::perception::PointPillarsPostprocessor>(m,
                                                                  "PointPillarsPostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::perception::PointPillarsPostprocessor& self,
              std::vector<FDTensor>& inputs) {
             std::vector<vision::PerceptionResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PointPillarsPostprocessor.");
             }
             return results;
           });

  pybind11::class_<vision::perception::PointPillars, FastDeployModel>(m, "PointPillars")
      .def(pybind11::init<std::string, std::string, RuntimeOption, ModelFormat>())
      .def("predict",
           [](vision::perception::PointPillars& self, 
              FDTensor& voxels, FDTensor& coords, FDTensor& num_points_per_voxel) {
             vision::PerceptionResult res;
             self.Predict(voxels, coords, num_points_per_voxel, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::perception::PointPillars& self,
              FDTensor& voxels_batch, FDTensor& coords_batch,
              FDTensor& num_points_per_voxel_batch) {
             std::vector<vision::PerceptionResult> results;
             self.BatchPredict(voxels_batch, coords_batch, num_points_per_voxel_batch, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::perception::PointPillars::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::perception::PointPillars::GetPostprocessor);
}
}  // namespace fastdeploy
