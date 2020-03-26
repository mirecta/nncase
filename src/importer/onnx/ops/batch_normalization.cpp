/* Copyright 2020 Miroslav Talasek <miroslav.talasek@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../onnx_importer.h"

#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/binary.h>
#include <hlir/ops/conv2d.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/unary.h>
#include <hlir/ops/reshape.h>
#include <hlir/op_utils.h>

// using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

/*
void printShape(const shape_t& shp, const std::string& name = "unknown"){

    printf("Shape of %s ->(",name.c_str());
    for (int i = 0; i < shp.size(); ++i){
        printf(" %d ", int(shp[i]));
    }
    printf(")\n");
}
*/

void onnx_importer::convert_op_BatchNormalization(const NodeProto& node)
{
    /// this is from pytorch 
    /// output(n, c, h, w)
    ///     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * scale(c)
    ///         + B(c)
    ///     = input(n, c, h, w) * inv_var(c) * scale(c)
    ///         - mean(c) * inv_var(c) * scale(c) + B(c),
    /// where inv_var(c) = 1 / sqrt(var(c) + eps).
    
    /// So the linear term for 1x1 convolution, weight(c) = inv_var(c) * scale(c),
    ///   the constant term bias(c) = B(c) - mean(c) * inv_var(c) * scale(c)
    /// Apply the linear terms to the input,
    /// output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
    /// 

    const auto &input { node.input()[0] };
    const auto &input_scale { node.input()[1] };
    const auto &input_B { node.input()[2] };
    const auto &input_mean { node.input()[3] };
    const auto &input_var { node.input()[4] };

    const auto &output { node.output()[0] };

    const auto &input_shape { get_shape(input) };
    
    const auto opt_eps { get_attribute<float>(node, "epsilon").value() };

    const auto* scale_initializer { get_initializer(input_scale) };
    if (!scale_initializer)
        throw std::runtime_error("Can't find initializer for scale input");
    auto&& scale_value { to<xt::xarray<float>>(*scale_initializer) };
    
    const auto* B_initializer { get_initializer(input_B) };
    if (!B_initializer)
        throw std::runtime_error("Can't find initializer for B input");
    auto&& B_value { to<xt::xarray<float>>(*B_initializer) };

    const auto* mean_initializer { get_initializer(input_mean) };
    if (!mean_initializer)
        throw std::runtime_error("Can't find initializer for mean input");
    auto&& mean_value { to<xt::xarray<float>>(*mean_initializer) };

    const auto* var_initializer { get_initializer(input_var) };
    if (!var_initializer)
        throw std::runtime_error("Can't find initializer for var input");
    auto&& var_value { to<xt::xarray<float>>(*var_initializer) };

    //compute new shape for weight and for bias    
    xt::xarray<float>::shape_type wshape = {input_shape[1], 1, 1, 1};
    auto weights = xt::xarray<float>::from_shape(wshape);
    xt::xarray<float>::shape_type bshape = {input_shape[1]};
    auto bias = xt::xarray<float>::from_shape(bshape);
    double inv_var;
    
    //printShape(weights.shape(), "weights");
    //prepare weight and bias for 1x1 convolution (we can use hardware KPU :D )
    //we must do it manual 
    //weight as convolution kernel 1x1
    for(int i=0; i < input_shape[1]; ++i){
        inv_var = 1. / sqrt(var_value[i] + opt_eps);
        weights.data()[i] = inv_var * scale_value[i]; 
        bias.data()[i] = B_value[i] - mean_value[i] * inv_var * scale_value[i];
    }
    //now we ready for conv2d 1x1 + bias op 
    auto conv2d_op {graph_.emplace<conv2d>(move(input_shape), move(weights), move(bias), 1, padding { 0, 0 }, padding { 0, 0 },
                        1, 1, 1, 1, value_range<float>::full())};

    //printShape(conv2d_op->output().shape(), "out");
    input_tensors_.emplace(&conv2d_op->input(), input);
    output_tensors_.emplace(output, &conv2d_op->output());
   
}