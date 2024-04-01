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
"""Script wraps launch_training to run with accelerate for multi and single GPU cases.
Read accelerate_launch_args configuration via environment variable `SFT_TRAINER_CONFIG_JSON_PATH`
for the path to the JSON config file with parameters or `SFT_TRAINER_CONFIG_JSON_ENV_VAR`
for the encoded config string to parse.
"""

# Standard
import json
import os
import base64
import pickle
import logging

# Third Party
from accelerate.commands.launch import launch_command

# Local
from tuning.utils.config_utils import process_accelerate_launch_args


def txt_to_obj(txt):
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)


def main():
    LOGLEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL)

    json_path = os.getenv("SFT_TRAINER_CONFIG_JSON_PATH")
    json_env_var = os.getenv("SFT_TRAINER_CONFIG_JSON_ENV_VAR")

    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            json_config = json.load(f)
    elif json_env_var:
        json_config = txt_to_obj(json_env_var)
    else:
        raise ValueError(
            "Must set environment variable 'SFT_TRAINER_CONFIG_JSON_PATH' \
        or 'SFT_TRAINER_CONFIG_JSON_ENV_VAR'."
        )

    logging.debug("Json config parsed: %s", json_config)

    args = process_accelerate_launch_args(json_config)

    launch_command(args)


if __name__ == "__main__":
    main()
