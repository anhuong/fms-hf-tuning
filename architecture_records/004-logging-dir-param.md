# Adding Logging Dir to Stream Loss Metrics

**Deciders(s)**:  Anh Uong (anh.uong@ibm.com), Sukriti Sharma (sukriti.sharma4@ibm.com)
**Date (YYYY-MM-DD)**:  2024-07-10
**Obsoletes ADRs**:  
**Modified By ADRs**:  
**Relevant Issues**: 

- [Summary and Objective](#summary-and-objective)
  - [Motivation](#motivation)
  - [User Benefit](#user-benefit)
- [Decision](#decision)
  - [Alternatives Considered](#alternatives-considered)
- [Consequences](#consequences)
- [Detailed Design](#detailed-design)

## Summary and Objective

There is a need for the continuous streaming to the log file to be accessible during tuning in order to display the information in a graph. Currently, although loss is continuously streamed to the ttraining_logs.jsonl`, this file is not accessible in the given `output_dir` specified by the user until tuning is complete.


### Motivation

Tuning creates many checkpoints but in order for products to have the direct path to the tuned model/prompt, we only mount the final checkpoint to the user specified `output_dir`. In this way, products can use the same given `output_dir` as the `MODEL_NAME` for inference with TGIS. In the future, we will explore a solution for mounting multiple checkpoints and allowing the user to specify a specific checkpoint which can be appended to the `output_dir` to give the full path to the model.

In the launch_training script we overwrite the `output_dir` with a temp dir so that all of the checkpoints aren't written to the `output_dir` until tuning is complete . We copy over both the final checkpoint and the training logs to the user set `output_dir` at the end when tuning is complete.

Thus we  introduce `logging_dir` only on library side where loss function logs will be written. It will be an optional field and set to `output_dir` if not specified. This parameter need not be exposed up through KFTO stack.

- This will cause final output directory to be created with loss file while tuning.
- Model checkpoints will be in tmpdir.
- At end of tuning, we also copy last checkpoint to final output directory.
 

### User Benefit

Enables user to have loss streamed to training logs during tuning and not have to wait until tuning is complete for the data.

## Decision

Currently we set the file where training logs and metrics are outputted using [`FileLoggingTrackerConfig.training_logs_filename`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/bf22a2f91f06d8cb4d50c925ec83b3f24337c9f3/tuning/config/tracker_configs.py#L20-L21) which is set to `training_logs.jsonl` by default. This is used in tandem with [FileLoggingTracker](https://github.com/foundation-model-stack/fms-hf-tuning/blob/bf22a2f91f06d8cb4d50c925ec83b3f24337c9f3/tuning/trackers/filelogging_tracker.py#L32) and the `FileLoggingCallback` to record designated metrics to the corresponding log file. This may include training loss, validation loss, and additional metrics specified by the user. 

The metrics are continually written to the training logs file already, which works well if one is using the library or calling `sft_trainer.py` directly. However as mentioned above, when using the image with `accelerate_launch.py`, a temp dir is used to store all of the checkpoints and the training logs and only after tuning is completed is the file written to the designated `output_dir`.

A new flag `--logging_dir` can be added to the existing `TrainingArguments`, however, `transformers.TrainingArguments` already has a [logging_dir flag](https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/training_args.py#L307-L309) used to export Tensorboard metrics. This is only enabled when Tensorboard is installed. Transformers `logging_dir` is set by default to `output_dir/runs/CURRENT_DATETIME_HOSTNAME`. Thus when the flags are parsed within our [sft_trainer.py](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py#L409), `logging_dir` is already set by default.

To overcome this and set `logging_dir` to the `output_dir` by default, we can check:
```
if "/runs/" in logging_dir:
   logging_dir = output_dir
```
so that logging_dir is not set by default to the output_dir.

### Alternatives Considered

- We can't use a different flag like `log_dir` as the flag must exist in `transformers.TrainingArguments` in order for the `FileLoggingCallback` to work as `FileLoggingCallback` only has `transformers.TrainingArguments` accessible when [writing logs](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/trackers/filelogging_tracker.py#L37).

## Consequences

- Replaces logging_dir default utiltiy from within `transformers.TrainingArguments`
- If tensorboard is installed, will write tensorboard metrics out to `output_dir`


## Detailed Design

This section is optional. Elaborate on details if they’re important to understanding the design, but would make it hard to read the proposal section above.

