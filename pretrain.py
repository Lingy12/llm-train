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
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import wandb

import torch
import transformers
# import utils
import hashlib
import json
from torch.utils.data import Dataset
from transformers import Trainer
from modules.data import RawPretrainDataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

from modules.logger_utils import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PREFIX_CHECKPOINT_DIR='checkpoint'

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_peft: Optional[bool] = field(default=False)
    lora_trainable: Optional[str] = field(default='embed_tokens,lm_head,q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj')
    modules_to_save: Optional[str] = field(default='embed_tokens,lm_head')
    lora_rank:  Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.05)
    
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # max_length=tokenizer.model_max_length,
            truncation=False,
        )
        for text in strings
    ]
    raw_input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # split input_ids with model_max_length
    input_ids = []
    labels = []
    max_length = tokenizer.model_max_length
    for instance_ids in raw_input_ids:
        for i in range(0, len(instance_ids), max_length):
            input_ids.append(instance_ids[i : i + max_length])
            labels.append(instance_ids[i : i + max_length])
            # if len(instance_ids[i : i + max_length]) < max_length:
            #     logger.warning('Warning: len(instance_ids[i : i + max_length]) < max_length')
            #     logger.warning(f"len(instance_ids[i : i + max_length]) < max_length: {len(instance_ids[i : i + max_length])} < {max_length}")

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def preprocess(
    examples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    print('block num', len(input_ids))
    print(input_ids[0].shape, input_ids[1].shape)
    # count all lengths
    cnt = sum([input_id.shape[0] for input_id in input_ids])
    print('# of tokens', cnt)
    labels = copy.deepcopy(input_ids)
    return dict(input_ids=input_ids, labels=labels)


class PretrainDataset(Dataset):
    """Dataset for pretraining."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length):
        super(PretrainDataset, self).__init__()
        self.raw_data = RawPretrainDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)
        files_ls = self.raw_data.files_ls

        # Hash the file list for caching
        file_ls_name=f'{os.path.basename(data_path)}'
        # if os.path.exists('.cache')
        os.makedirs('.cache', exist_ok=True)
        cache_path = f".cache/{file_ls_name}_cached.pt"
        
        if os.path.exists(cache_path):
            logger.warning(f"Loading cached data from {cache_path}")
            self.data_dict = torch.load(cache_path)
        else:
            logger.warning("Tokenizing inputs... This may take some time...")
            self.data_dict = preprocess(files_ls, tokenizer)
            torch.save(self.data_dict, cache_path)
            logger.warning(f"Cached tokenized data at {cache_path}")

        self.input_ids = self.data_dict["input_ids"]
        self.labels = self.data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForPretrainDataset(object):
    """Collate examples for pretraining."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for pretraining."""
    train_dataset = PretrainDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_length=training_args.model_max_length)
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True, 
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True, 
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    if model_args.use_peft:
        logger.info("Init new peft model")
        target_modules = model_args.lora_trainable.split(',')
        modules_to_save = model_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    data_module = make_pretrain_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=os.path.join(training_args.output_dir, 'checkpoint-final'))


if __name__ == "__main__":
    train()