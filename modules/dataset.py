from torch.utils.data import Dataset
from transformers import BertTokenizer


class DatasetQA(Dataset):
    def __init__(self, src, trg, mode: str = "bert-base-uncased", max_length: int = 512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(mode)

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src = src
        self.max_length = max_length
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        src = str(self.src[item])
        trg = str(self.trg[item]['text'][0])
        enc_src = self.tokenizer.encode_plus(
            text=src,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            # return_length=True,
            pad_to_max_length=True

        )
        enc_trg = self.tokenizer.encode_plus(
            text=trg,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            # return_length=True,
            pad_to_max_length=True,
            truncation=True

        )
        return enc_src['input_ids'], enc_trg['input_ids']
