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
from dataclasses import asdict
import logging
import os

# Third Party
from accelerate.commands.launch import launch_command_parser
from peft import LoraConfig, PromptTuningConfig
import torch

# Local
from tuning.config import peft_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")


def create_tuning_config(peft_method, **kwargs):
    """Create peft_config Tuning config
    Args:
        peft_method: str
           lora, pt or None
        kawrgs: parameters to initialize library configs with
     Return:
        peft_config.LoraConfig | peft_config.PromptTuningConfig | None
    """
    assert peft_method in [
        None,
        "lora",
        "pt",
        "None",
    ], f"peft config {peft_method} not defined in peft.py"
    if peft_method == "lora":
        tune_config = peft_config.LoraConfig()
        update_config(tune_config, **kwargs)
    elif peft_method == "pt":
        tune_config = peft_config.PromptTuningConfig()
        update_config(tune_config, **kwargs)
    else:
        tune_config = None  # full parameter tuning
    return tune_config


def get_hf_peft_config(task_type, tuning_config):
    """Return HF PEFT config for tuning based on type of tuning config passed
    Args:
        task_type: str
        tuning_config: peft_config.LoraConfig | peft_config.PromptTuningConfig | None
    Return: HF PEFT config or None
    """
    if isinstance(tuning_config, peft_config.LoraConfig):
        lora_config = asdict(tuning_config)
        if lora_config["target_modules"] == ["all-linear"]:
            lora_config["target_modules"] = "all-linear"
        hf_peft_config = LoraConfig(task_type=task_type, **lora_config)
    elif isinstance(tuning_config, peft_config.PromptTuningConfig):
        hf_peft_config = PromptTuningConfig(
            task_type=task_type, **asdict(tuning_config)
        )
    else:
        hf_peft_config = None  # full parameter tuning

    return hf_peft_config


def process_accelerate_launch_args(job_config):
    parser = launch_command_parser()
    # Map to determine which params are flags ie. don't require a value to be set
    actions_type_map = {
        action.dest: type(action).__name__ for action in parser._actions
    }

    # Parse accelerate_launch_args
    accelerate_launch_args = []
    accelerate_config = job_config.get("accelerate_launch_args", {})
    if accelerate_config:
        logging.info("Using accelerate_launch_args configs: %s", accelerate_config)
        for key, val in accelerate_config.items():
            if actions_type_map.get(key) == "_AppendAction":
                for param_val in val:
                    accelerate_launch_args.extend([f"--{key}", str(param_val)])
            elif (actions_type_map.get(key) == "_StoreTrueAction" and val) or (
                actions_type_map.get(key) == "_StoreFalseAction" and not val
            ):
                accelerate_launch_args.append(f"--{key}")
            else:
                accelerate_launch_args.append(f"--{key}")
                # Only need to add key for params that aren't flags ie. --quiet
                if actions_type_map.get(key) == "_StoreAction":
                    accelerate_launch_args.append(str(val))

    if job_config.get("multi_gpu"):
        # Add FSDP config
        fsdp_filepath = accelerate_config.get("config_file") or os.getenv(
            "FSDP_DEFAULTS_FILE_PATH", "/app/accelerate_fsdp_defaults.yaml"
        )
        if os.path.exists(fsdp_filepath):
            logging.info("Using accelerate config file: %s", fsdp_filepath)
            accelerate_launch_args.extend(["--config_file", fsdp_filepath])

        if not accelerate_config.get("num_processes"):
            num_gpus = torch.cuda.device_count()
            logging.info("Using num_processes: %s for accelerate launch", num_gpus)
            accelerate_launch_args.extend(["--num_processes", str(num_gpus)])
    else:
        logging.info(
            "Passing num_processes:1 for accelerate launch. To enable multiple gpus enable \
            `multi_gpu` flag and specify number of gpus using `num_processes` param, otherwise \
            torch.cuda.device_count() will be used to deduce the number of processes to use."
        )
        accelerate_launch_args.extend(["--num_processes", str(1)])

    # Add training_script
    accelerate_launch_args.append("/app/launch_training.py")

    logging.debug("accelerate_launch_args: %s", accelerate_launch_args)
    args = parser.parse_args(args=accelerate_launch_args)
    logging.debug("accelerate launch parsed args: %s", args)
    return args
