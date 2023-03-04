from logging import getLogger
from typing import Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = getLogger(__name__)


class Tokens:
    eos = '<|endoftext|>'
    pad = '<|pad|>'
    sos = '<|startoftext|>'


class DatasetLLama(Dataset, Tokens):
    def __init__(self, data: Optional[List[str]],
                 tokenizer: Optional[transformers.GPT2Tokenizer], max_length: Optional[int] = 768):
        self.tokenizer = tokenizer

        self.input_ids = []
        self.max_length = max_length
        logger.info('Tokenizing Data')
        pbar = tqdm(enumerate(data))
        failed = 0
        for i, txt in pbar:
            if txt != '' and not txt.startswith(' ='):
                pbar.set_postfix(failed=failed, collected=i + 1 - failed)

                encodings_dict = tokenizer(self.sos + txt + self.eos, truncation=True,
                                           max_length=max_length, padding="do_not_pad")

                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            else:
                failed += 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

    def encode(self, text):
        enc_trg = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_length,
            padding='do_not_pad',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return enc_trg
