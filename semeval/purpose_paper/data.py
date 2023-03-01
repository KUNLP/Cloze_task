from typing import Optional, List, Union
import datasets
from dataclasses import dataclass

import torch
from torch.utils.data import (
    DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
)
import pytorch_lightning as pl

from transformers import (
    AutoTokenizer
)
import pandas as pd
import argparse
from purpose_paper.utils import convert_examples_to_features, load_labelset
from purpose_paper.define_class import InputExample, InputFeature



class SemEvalDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.model_name_or_path = self.args.model_name_or_path
        self.max_seq_length = self.args.max_seq_length
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size
        self.test_batch_size = self.args.test_batch_size
        self.num_labels = self.args.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def _create_examples(self, dataset: pd.DataFrame):
        examples = []
        dataset = dataset.fillna("")
        for _, row in dataset.iterrows():
            for answer in range(1,6):
                examples.append(
                    InputExample(
                        id=f"{row['Id']}_{answer}",
                        title=row['Article title'],
                        prev_sentence=row['Previous context'],
                        now_sentence=row['Sentence'],
                        next_sentence=row['Follow-up context'],
                        answer=row[f"Filler{answer}"]
                    )
                )
        return examples

    def _convert_features(self, examples: List[InputExample]):
        return convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            args=self.args
        )

    def _create_dataset(self, dataset: pd.DataFrame, stage: Optional[str]):
        examples = self._create_examples(dataset)
        features = self._convert_features(examples)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )

        if 'train' in stage:
            labelset = load_labelset('train')
            all_label = torch.tensor(labelset, dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_label
            )
            return dataset
        elif 'val' in stage:
            labelset = load_labelset('val')
            all_label = torch.tensor(labelset, dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_label
            )
            return dataset
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids
        )
        return dataset

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            train_dataset = pd.read_csv('../data/train/traindata.tsv', sep='\t', quoting=3)
            self.train_data = self._create_dataset(train_dataset, 'train')
            val_dataset = pd.read_csv('../data/dev/devdata.tsv', sep='\t', quoting=3)
            self.val_data = self._create_dataset(val_dataset, 'val')

        if stage in (None, 'test'):
            test_dataset = pd.read_csv('../data/test/testdata.tsv', sep='\t', quoting=3)
            self.test_data = self._create_dataset(test_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False)
