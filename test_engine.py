# from erutils.utils import read_yaml, read_json
# from transformers import GPT2Config, GPT2Model, TrainingArguments, Trainer, LineByLineTextDataset, BertTokenizer
#
# from utils.utils import DatasetQA
#
# config_path = 'config/PTTGenerative-small.yaml'
#
# if __name__ == "__main__":
#     cfg = read_yaml(config_path)
#
#     data_path = cfg['data_path']
#     epochs = cfg['epochs']
#     lr = float(cfg['lr'])
#     max_length = cfg['chunk']
#     number_of_heads = cfg['number_of_heads']
#     number_of_layers = cfg['number_of_layers']
#     embedded = cfg['embedded']
#     use_train = cfg['train']
#     batch_size = cfg['batch_size']
#     # ssm = SummaryWriter(log_dir='results/out')
#     data = read_json(data_path)
#
#     questions = [data[v]['question'] for v in data]
#     answers = [data[v]['answer'] for v in data]
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", max_len=max_length, padding='longest')
#     datas = DatasetQA(
#         src=questions,
#         trg=answers,
#         max_length=max_length
#     )
#     # tokenizer.encode_plus
#     dataset_ll = LineByLineTextDataset(file_path='data/q&a_cleaned.txt', tokenizer=tokenizer,
#                                        block_size=max_length)
#     vocab_size: int = tokenizer.vocab_size
#     pad_token: int = tokenizer.pad_token_id
#     eos_token: int = tokenizer.eos_token_id
#     bos_token: int = tokenizer.bos_token_id
#
#     config = GPT2Config(eos_token_id=eos_token, bos_token_id=bos_token, pad_token_id=pad_token, n_head=number_of_heads,
#                         n_layer=number_of_layers, n_embd=embedded, chunk_size_feed_forward=max_length,
#                         vocab_size=vocab_size)
#
#     model = GPT2Model(config=config)
#     print(model)
#     print(sum(p.numel() for p in model.parameters()) / 1e6, ' Million Parameters ')
#     training_arg = TrainingArguments(
#         output_dir='./GPT2',
#         save_steps=100,
#         save_total_limit=2,
#         overwrite_output_dir=True,
#         prediction_loss_only=True,
#         auto_find_batch_size=True,
#         num_train_epochs=5
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_arg,
#         train_dataset=dataset_ll,
#
#     )
#
#     trainer.train()
#     trainer.save_model('./GPT2')
