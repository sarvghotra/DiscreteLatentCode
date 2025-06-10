import json
import os
import re
import urllib.request
import zipfile
from itertools import chain

import numpy as np
import requests
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import GPT2TokenizerFast


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


class SEMDataset(Dataset):
    def __init__(self, sem_dir):
        self.sem_dir = sem_dir
        self.sem_files = sorted(os.listdir(sem_dir))

    def __len__(self):
        return len(self.sem_files)

    def __getitem__(self, idx):
        sem_file = self.sem_files[idx]
        sem = np.load(os.path.join(self.sem_dir, sem_file))
        sem = torch.from_numpy(sem)
        return sem


def collate_tensor_fn(batch):
    l = [b["labels"] for b in batch]
    l = torch.tensor(l)
    return l


def get_dataset(name, mode, hf_dataset, cache_dir=None, block_size=1024, num_proc=8):
    sem_dir = <sem_dir>
    if hf_dataset:
        dataset = load_from_disk(name)
    else:
        dataset = SEMDataset(sem_dir)
    return dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}."
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}."
        )

    train_set = get_dataset(
        config.data.train,
        "train",
        config.data.hf_dataset,
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
    )
    valid_set = get_dataset(
        config.data.valid,
        "validation" if config.data.valid != "text8" else "test",
        config.data.hf_dataset,
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
    )

    if distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None

    if config.data.hf_dataset:
        collate_fn = collate_tensor_fn
    else:
        collate_fn = None
    train_loader = cycle_loader(
        DataLoader(
            train_set,
            batch_size=config.training.batch_size
            // (config.ngpus * config.training.accum),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=(train_sampler is None),
            persistent_workers=True,
        )
    )
    valid_loader = cycle_loader(
        DataLoader(
            valid_set,
            batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )
    )
    return train_loader, valid_loader
