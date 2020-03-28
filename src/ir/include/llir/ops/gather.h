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
#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>

namespace nncase
{
namespace llir
{
    class gather : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_gather);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        int32_t axis() const noexcept { return axis_; }
        const xt::xarray<int32_t> &indices() const noexcept { return indices_; }
        gather(shape_t input_shape, xt::xarray<int32_t> indices, int32_t axis);

    private:
        xt::xarray<int32_t> indices_;
        int32_t axis_;
        int32_t to_copy_;
        int32_t loops_;
        int32_t mult_;

    };
}
}
