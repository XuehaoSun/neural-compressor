#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unary operator."""

from neural_compressor.adaptor.ox_utils.operators.ops import op_registry, Operator

@op_registry(op_types="Abs, Exp, Log, Round, Sqrt")
class UnaryOperator(Operator):
    """Unary operator."""
    
    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(UnaryOperator, self).__init__(onnx_quantizer, onnx_node)