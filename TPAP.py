import pandas as pd
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from torch.utils.data import Dataset
import torch

# from modules.modeling_t5 import T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
data_frame = pd.read_csv('ipynb/news_summary.csv')
data_frame["text"] = "summarize: " + data_frame["text"]

kwargs = {"d_ff": 512,
          "d_kv": 64,
          "d_model": 192,
          "decoder_start_token_id": 0,
          "dropout_rate": 0.1,
          "eos_token_id": 1,
          "feed_forward_proj": "gated-gelu",
          "initializer_factor": 1.0,
          "is_encoder_decoder": True,
          "layer_norm_epsilon": 1e-06,
          "model_type": "t5",
          "n_positions": 512,
          "num_decoder_layers": 12,
          "num_heads": 12,
          "num_layers": 2,
          "output_past": True,
          "pad_token_id": 0,
          "relative_attention_max_distance": 128,
          "relative_attention_num_buckets": 32,
          "tie_word_embeddings": False,
          "transformers_version": "4.23.1",
          "use_cache": True,
          "vocab_size": 32128}
config = T5Config(max_length=512, **kwargs)

model = T5ForConditionalGeneration(config=config)
print(sum(m.numel() for m in model.parameters()) / 1e6)


class DatasetC(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.summ_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids']
        source_mask = source['attention_mask']
        target_ids = target['input_ids']
        target_mask = target['attention_mask']

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


source_length = config.max_length
target_length = config.max_length
device = 'cpu'
dataset = DatasetC(dataframe=data_frame, tokenizer=tokenizer, source_len=source_length, target_len=target_length,
                   source_text='text', target_text='headlines')
data = dataset.__getitem__(1)

y = data['target_ids'].to(device, dtype=torch.long)
decoder_input = y[:, :-1].contiguous()
lm_labels = y[:, 1:].clone().detach()
lm_labels[y[:, 1:] == 0] = -100

input_id = data['source_ids'].to(device, dtype=torch.long)
mask = data['source_mask'].to(device, dtype=torch.long)
if __name__ == '__main__':
    output = model(
        input_ids=input_id,
        attention_mask=mask,
        decoder_input_ids=decoder_input,
        labels=lm_labels
    )

    # print(output)
