# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F


def pad(self, pad, mode="constant", value=None):
    # The Ascend dispatcher excludes pad. Use the native implementation for
    # direct flag_gems.pad calls as well, including empty and large tensors.
    return F.pad(self, pad, mode=mode, value=value)


def constant_pad_nd(self, pad_list, value=0):
    return pad(self, pad_list, mode="constant", value=value)
