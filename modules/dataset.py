from torch.utils.data import Dataset
import torch
from typing import Optional, List
from logging import getLogger
from tqdm.auto import tqdm

logger = getLogger(__name__)


class Tokens:
    eos = '<|endoftext|>'
    pad = '<|pad|>'
    sos = '<|startoftext|>'


class DatasetLLama(Dataset, Tokens):
    def __init__(self, txt_list: Optional[List[str]],
                 tokenizer, max_length: Optional[int] = 768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.max_length = max_length
        logger.info('Tokenizing Data')
        for txt in tqdm(txt_list):
            encodings_dict = tokenizer(self.sos + txt + self.eos, truncation=True,
                                       max_length=max_length, padding="do_not_pad")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

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
