#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import sys
sys.path.append('..')
from modules.logger_utils import get_logger
import logging
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from datasets import load_from_disk


import torch
import transformers
# import utils
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

logger = get_logger(__name__)

def extract_all_text(hf_ds):
    sentences_list = hf_ds['train']['text']
    return sentences_list

class RawPretrainDataset(Dataset):
    """Raw Dataset for llm pretraining."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super(RawPretrainDataset, self).__init__()
        logger.warning("Loading data...")

        self.files_ls = []
        # load pretrain markdown data from data_path (read all .md files in data_path)
        for file in os.listdir(data_path):
            real_path = os.path.join(data_path, file)
            if file.endswith(".md") or file.endswith(".txt"):
                pretrain_data_path = data_path + '/' + file
                self.files_ls.extend(self.read_md(pretrain_data_path, tokenizer, max_length))
            if os.path.isdir(real_path):
                hf_ds = load_from_disk(real_path)
                self.files_ls.extend(extract_all_text(hf_ds))

        logger.warning("Loading data finished.")
        
        # shuffle files
        random_index = random.sample(range(len(self.files_ls)), len(self.files_ls))
        self.files_ls = [self.files_ls[i] for i in random_index]
        print('# of file: ', len(self.files_ls))
        logger.warning("# of file: " + str(len(self.files_ls)))
    
    def read_md(self, pretrain_data_path, tokenizer, max_length=512):
        logger.warning('Loading data from ' + pretrain_data_path)
        with open(pretrain_data_path, "r") as f:
            raw_file = f.read()

        # Split the text into segments of up to max_length, considering the EOS token
        segments = []
        i = 0
        while i < len(raw_file):
            # Determine the end index of the next segment
            end_index = i + max_length - 1
            if end_index >= len(raw_file):
                segments.append(raw_file[i:] + tokenizer.eos_token)
            else:
                # Ensure not to cut in the middle of a word
                if raw_file[end_index] != ' ' and end_index + 1 < len(raw_file) and raw_file[end_index + 1] != ' ':
                    # Move the end index to the end of the current word
                    while end_index > i and raw_file[end_index] != ' ':
                        end_index -= 1
                segments.append(raw_file[i:end_index + 1] + tokenizer.eos_token)
            i = end_index + 1

        return segments

