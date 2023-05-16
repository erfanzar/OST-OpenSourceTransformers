import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling


def configure_model(model_path: str,
                    tokenizer_path: str,
                    device_map: str | None = None,
                    torch_dtype: str | torch.dtype = torch.bfloat16,
                    from_safetensors: bool | None = True):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, torch_dtype=torch_dtype,
                                                 from_safetensors=from_safetensors)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def set_ff_model(_model, on_layers, embedding_requires_grad=True):
    mlp, ll, q_p, v_p, k_p, o_p = [], [], [], [], [], []

    for i, (name, param) in enumerate(_model.named_parameters()):
        for ll in on_layers:
            if f"{ll}" not in name:
                param.requires_grad = False

        if 'mlp' in name:
            param.requires_grad = False
        if i == 0 and embedding_requires_grad:
            param.requires_grad = True

    for i, (name, param) in enumerate(_model.named_parameters()):
        print("{:>5} : {:<60} : {:>15} => {:>25}".format(i, name, param.requires_grad, param.numel() / 1e6))
        if 'q_proj' in name:
            q_p.append(param.numel() / 1e6)
        if 'k_proj' in name:
            k_p.append(param.numel() / 1e6)
        if 'v_proj' in name:
            v_p.append(param.numel() / 1e6)
        if 'o_proj' in name:
            o_p.append(param.numel() / 1e6)
        if 'mlp' in name:
            mlp.append(param.numel() / 1e6)
        for ll in on_layers:
            if f"{ll}" in name:
                ll.append(param.numel() / 1e6)

    print(f'K Proj Contain {sum(k_p)} Million Parameters')
    print(f'Q Proj Contain {sum(q_p)} Million Parameters')
    print(f'O Proj Contain {sum(o_p)} Million Parameters')
    print(f'V Proj Contain {sum(v_p)} Million Parameters')
    print(f'MLP Contain    {sum(mlp)} Million Parameters')

    print(f'\n------\nEach Block Contain {sum(ll)} Million Parameters (Based On {on_layers} Block)')

    train_able_parameters = 0
    for i, (name, param) in enumerate(_model.named_parameters()):
        train_able_parameters += param.numel() / 1e6 if param.requires_grad else 0
    print(f'Total TrainAble Parameters In Model Is {train_able_parameters} Million Parameters')

    for i, (name, param) in enumerate(_model.named_parameters()):
        if param.requires_grad:
            print("{:>5} : {:<60} : {:>15} => {:>25}".format(i, name, param.requires_grad, param.numel() / 1e6))
    return _model


def make2d(tensor):
    return tensor.view(-1, tensor.size(-1))


def train(
        model, optimizer, data_loader, scheduler_lr=None, epochs: int = 5
):
    pbar = tqdm(range(epochs * len(data_loader)))
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['input_ids']
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            if scheduler_lr is not None:
                scheduler_lr.step()
                pbar.set_postfix(loss=loss.item(), learning_rate=scheduler_lr.get_lr())


def configure_data_loader(path: str = 'erfanzar/Data-LGeM-60K', field='train', batch_size: int = 4):
    dataset = load_dataset(path)

    def collect_fn(batch):
        holder_dict = {}
        for key in batch[0].keys():
            tensor = torch.stack([torch.tensor(stack[key]) for stack in batch])
            holder_dict[key] = tensor
        return holder_dict

    dataloader = DataLoader(dataset[field], batch_size=batch_size, shuffle=None, collate_fn=collect_fn)
    return dataloader


def main():
    epochs = 2
    lr = 7e-5
    weight_decay = 1e-2
    torch_dtype = torch.float32
    embedding_requires_grad = True
    batch_size = 4

    dataset_hub = 'erfanzar/Data-LGeM-60K-500'
    path_to_save = 'erfanzar/LGeM-M'

    model, tokenizer = configure_model(
        'erfanzar/LGeM-7B',
        'erfanzar/LGeM-M',
        torch_dtype=torch_dtype
    )
    model = set_ff_model(model, embedding_requires_grad=embedding_requires_grad)
    train_dataloader = configure_data_loader(dataset_hub, 'train', batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train(model, optimizer, train_dataloader, epochs=epochs)
    model.save_pretrained(path_to_save)


def transformer_trainer(model_path,
                        tokenizer_path,
                        per_gpu_eval_batch_size: int = 8,
                        num_train_epochs=2,
                        learning_rate=1e-4,
                        embedding_requires_grad=False,
                        dataset_hub='erfanzar/Data-LGeM-60K-500',
                        output_dir='erfanzar/LGeM-M',

                        ):
    model, tokenizer = configure_model(model_path=model_path, tokenizer_path=tokenizer_path)
    model = set_ff_model(model, [16, 15, 14, 13, 12], embedding_requires_grad=embedding_requires_grad)
    train_dataset = load_dataset(dataset_hub)
    args = TrainingArguments(
        per_gpu_eval_batch_size=per_gpu_eval_batch_size,
        weight_decay=0.01,
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        report_to='none',
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        fp16=True,
        do_eval=False,
        fsdp='auto_wrap full_shard',
        fsdp_transformer_layer_cls_to_wrap='GPTBlock',
        output_dir=output_dir,
        save_strategy='epoch',
        logging_dir=f"{output_dir}/logs",
    )
    trainer = Trainer(
        args=args,
        model=model,
        data_collator=DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer),
        train_dataset=train_dataset['train'],
        eval_dataset=None
    )
    trainer.train()


if __name__ == '__main__':
    transformer_trainer()
