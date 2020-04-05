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
#include <hlir/op_utils.h>
#include <hlir/ops/gather.h>
#include <xtensor/xarray.hpp>
#include <llir/ops/gather.h>

using namespace nncase;
using namespace nncase::hlir;

gather::gather(shape_t input_shape, xt::xarray<int32_t> indices, int32_t axis)
    :indices_(std::move(indices))
    ,axis_(axis)
    ,to_copy_(0)
    ,loops_(0)
    ,mult_(0)
{ 
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, hlir::get_gather_output_shape(input_shape,  indices.shape(), axis_, to_copy_, loops_, mult_));
}

void gather::compile(hlir_compile_context &context)
{
    auto l_c = context.graph.emplace<llir::gather>(input().shape(), indices_, axis_);
    context.add_input(input(), l_c->input());
    context.add_output(output(), l_c->output());
}
