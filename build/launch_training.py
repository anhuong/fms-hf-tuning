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
"""Script wraps SFT Trainer to run for Train Conductor.
Read SFTTrainer configuration via environment variable `SFT_TRAINER_CONFIG_JSON_PATH`
for the path to the JSON config file with parameters or `SFT_TRAINER_CONFIG_JSON_ENV_VAR`
for the encoded config string to parse.
"""

# Standard
import sys
import traceback

# Third Party
from huggingface_hub.utils._validators import HFValidationError
from torch.cuda import OutOfMemoryError

# First Party
import logging

# Local
from tuning import sft_trainer
from tuning.utils.config_utils import get_json_config
from tuning.config.tracker_configs import TrackerConfigFactory
from build.utils import (
    process_launch_training_args,
    write_termination_log,
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)


def main():

    try:
        job_config = get_json_config()
        logging.debug("Input params parsed: %s", job_config)

        (
            model_args,
            data_args,
            training_args,
            tune_config,
            merge_model,
            file_logger_config,
            aim_config,
        ) = process_launch_training_args(job_config)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)

    try:
        tracker_config_args = TrackerConfigFactory(
            file_logger_config=file_logger_config, aim_config=aim_config
        )
        sft_trainer.train(
            model_args=model_args,
            data_args=data_args,
            train_args=training_args,
            peft_config=tune_config,
            tracker_configs=tracker_config_args,
        )
    except (MemoryError, OutOfMemoryError) as e:
        logging.error(traceback.format_exc())
        write_termination_log(f"OOM error during training. {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)
    except FileNotFoundError as e:
        logging.error(traceback.format_exc())
        write_termination_log("Unable to load file: {}".format(e))
        sys.exit(USER_ERROR_EXIT_CODE)
    except HFValidationError as e:
        logging.error(traceback.format_exc())
        write_termination_log(
            f"There may be a problem with loading the model. Exception: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except (TypeError, ValueError, EnvironmentError) as e:
        logging.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        write_termination_log(f"Unhandled exception during training: {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)

    return 0


if __name__ == "__main__":
    main()