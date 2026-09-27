import os
import random
import time
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.datasets.utils import Split
from megatron.core.datasets.utils_s3 import S3Config, is_s3_path
from megatron.core.utils import log_single_rank
from megatron.training import print_rank_0, get_args
from megatron.core import parallel_state
from megatron.legacy.data.dataset_utils import get_train_valid_test_split_
from megatron.core.datasets.utils import get_blend_from_list

from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import DecoderPackedMTFDataset, _build_train_valid_test_datasets
from mindspeed_llm.training.tokenizer import build_tokenizer
from mindspeed_llm.tasks.utils.error_utils import check_equal
from mindspeed_llm.tasks.preprocess.mtf_dataset import MTFDataset, get_packed_indexed_dataset
from mindspeed_llm.tasks.preprocess.templates import get_model_template


def build_blended_mtf_dataset(
    data_prefix,
    splits_string,
    seq_length: int,
    train_valid_test_num_samples,
    seed,
):
    """Build train, valid, and test datasets."""
    args = get_args()

    if len(data_prefix) == 1:
        data_prefix = data_prefix[0].split(',')
    blend = get_blend_from_list(data_prefix)
    paths, weights = blend

    tokenizer = build_tokenizer(args)
    pad_token = tokenizer.pad
    eos_token = tokenizer.eos

    all_train_datasets, all_valid_datasets, all_test_datasets = [], [], []

    for path in paths:
        train_dataset, valid_dataset, test_dataset = _build_train_valid_test_datasets(
            data_prefix=path,
            splits_string=splits_string,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed,
        )
        all_train_datasets.append(train_dataset)
        all_valid_datasets.append(valid_dataset)
        all_test_datasets.append(test_dataset)

    blended_train_datasets = BlendedMTFDataset(weights, all_train_datasets)
    blended_valid_datasets = BlendedMTFDataset(weights, all_valid_datasets)
    blended_test_datasets = BlendedMTFDataset(weights, all_test_datasets)

    return blended_train_datasets, blended_valid_datasets, blended_test_datasets


class BlendedMTFDataset(torch.utils.data.Dataset):
    def __init__(self, weights, datasets):
        self.weights = [len(dataset.mtf_dataset) if dataset else 1 for dataset in datasets] if not weights else weights
        print_rank_0(self.weights)
        self.datasets = datasets
        self.num_samples = sum([len(dataset) for dataset in self.datasets if dataset])
        self.num_datasets = len(self.datasets)
        self.dataset_index_map = self._build_dataset_index_map()

    def _build_dataset_index_map(self):
        error_threshold = -len(self)
        total_weight = sum(self.weights)
        normal_weight = [w / total_weight for w in self.weights]
        dataset_index_map = []
        data_sampled = [0 for _ in range(self.num_datasets)]
        for i in range(len(self)):
            max_error_index = -1
            max_error = error_threshold
            for j in range(self.num_datasets):
                if data_sampled[j] < len(self.datasets[j]):
                    error = normal_weight[j] * i - data_sampled[j]
                else:
                    error = error_threshold
                if error >= max_error:
                    max_error_index = j
                    max_error = error
            dataset_index_map.append((max_error_index, data_sampled[max_error_index]))
            data_sampled[max_error_index] += 1
        return dataset_index_map

    def __getitem__(self, idx):
        idx = idx % len(self)
        dataset_id, dataset_sample_id = self.dataset_index_map[idx]
        return {
            "dataset_id": dataset_id,
            **self.datasets[dataset_id][dataset_sample_id],
        }

    def __len__(self):
        return self.num_samples