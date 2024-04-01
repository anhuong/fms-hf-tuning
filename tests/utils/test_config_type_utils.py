# Copyright The FMS HF Tuning Authors
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

# Standard
import copy
import json
import os
from unittest.mock import patch

# Third Party
import pytest

# Local
from tuning.utils.config_utils import process_accelerate_launch_args

HAPPY_PATH_DUMMY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dummy_job_config.json"
)


# Note: job_config dict gets modified during post_process_job_config
@pytest.fixture(scope="session")
def job_config():
    with open(HAPPY_PATH_DUMMY_CONFIG_PATH, "r", encoding="utf-8") as f:
        dummy_job_config_dict = json.load(f)
    return dummy_job_config_dict


def test_process_accelerate_launch_args(job_config):
    job_config_copy = copy.deepcopy(job_config)
    args = process_accelerate_launch_args(job_config_copy)
    assert args.config_file == "fixtures/accelerate_fsdp_defaults.yaml"
    assert args.use_fsdp == True
    assert args.tpu_use_cluster == False


@patch("torch.cuda.device_count")
@patch("os.path.exists")
def test_process_accelerate_launch_args_gpu(patch_path_exisits, patch_cuda_device_num):
    patch_path_exisits.return_value = True
    patch_cuda_device_num.return_value = 5

    # When user passes custom fsdp config file but multi_gpu is not enabled, use custom config and assume single gpu
    temp_job_config = {
        "accelerate_launch_args": {"config_file": "dummy_fsdp_config.yaml"}
    }
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == "dummy_fsdp_config.yaml"
    assert args.num_processes == 1

    # When user passes custom fsdp config file and multi_gpu is enabled, use custom config and
    # torch.cuda.device_count to deduce num_processes
    temp_job_config = {
        "accelerate_launch_args": {"config_file": "dummy_fsdp_config.yaml"},
        "multi_gpu": True,
    }
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == "dummy_fsdp_config.yaml"
    assert args.num_processes == 5

    # When user passes custom fsdp config file, multi_gpu is enabled and num of gpus is also specified,
    # config file and num_processes should be overwritten with values provided by user
    temp_job_config = {
        "accelerate_launch_args": {
            "config_file": "dummy_fsdp_config.yaml",
            "num_processes": 3,
        },
        "multi_gpu": True,
    }
    args = process_accelerate_launch_args(temp_job_config)
    assert args.config_file == "dummy_fsdp_config.yaml"
    assert args.num_processes == 3
