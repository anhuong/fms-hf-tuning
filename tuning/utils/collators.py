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


class VisionDataCollator:
    def __init__(self, processor, text_field_name, image_field_name):
        self.processor = processor
        self.text_field_name = text_field_name
        self.image_field_name = image_field_name

    def __call__(self, batch):
        """
        Processes the batch containing text and images by adding labels and perform masking
        This collator takes a batch and returns a
        processed batch with input_ids and labels
        """

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        # TOOD: should we be ensuring EOS tokens is set?
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch
