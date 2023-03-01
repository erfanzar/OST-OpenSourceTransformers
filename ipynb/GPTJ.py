import logging
from logging import getLogger
from typing import Optional, List

import erutils
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, GPTJForCausalLM, GPTJConfig

level = logging.INFO
logger = getLogger(__name__)
logger.level = level


class Tokens(object):
    eos = '<|endoftext|>'
    pad = '<|pad|>'
    sos = '<|startoftext|>'


class GPTDataset(Dataset, Tokens):

    def __init__(self, txt_list: Optional[List[str]], tokenizer, max_length: Optional[int] = 768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer(self.sos + txt + self.eos, truncation=True,
                                       max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

    def encode(self, text):
        enc_trg = self.tokenizer.encode_plus(
            text=text,
            max_length=self.chunk,
            padding='do_not_pad',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return enc_trg


def main(mxl: Optional[int] = 256):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=Tokens.sos, eos_token=Tokens.eos, pad_token=Tokens.pad)
    model = GPTJForCausalLM(
        config=GPTJConfig(vocab_size=tokenizer.vocab_size + 3, bos_token_id=tokenizer.bos_token_id, max_length=mxl,
                          eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, n_positions=mxl,
                          n_layer=8, n_head=6, n_embd=600))

    logger.debug(f'{model}')

    data = open('../data/PGT-DATA-V2.txt', 'r', encoding='utf8').read()
    data_list = data.split(Tokens.eos)

    dataset = GPTDataset(data_list, tokenizer, mxl)

    batch = 2

    loader = DataLoader(dataset=dataset, batch_size=batch)
    epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 4e-4)

    erutils.fprint(f" MODEL PARAMETERS : {(sum(p.numel() for p in model.parameters()) / 1e6)}")

    for epoch in range(epochs):

        with tqdm(loader, total=dataset.__len__() // batch, colour='white') as progress_bar:
            for batch in progress_bar:
                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                optimizer.zero_grad()

                outputs = model(b_input_ids,
                                labels=b_labels,
                                attention_mask=b_masks,
                                token_type_ids=None
                                )
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                # progress_bar.set_description(f'loss : {loss.item()}')
                progress_bar.set_postfix(loss=loss.item(), epoch=epoch)


if __name__ == "__main__":
    main()
