import accelerate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import os
from torch.nn.parallel import DistributedDataParallel as DDP

LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))


def print_rank_0(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)


def configure_model(model_path: str,
                    tokenizer_path: str,
                    device_map: str | None = None,
                    torch_dtype: str | torch.dtype = torch.float32,
                    from_safetensors: bool | None = True,
                    move_model: bool | None = False):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, torch_dtype=torch_dtype,
                                                 low_cpu_mem_usage=True
                                                 )
    if move_model:
        print(f'MOVING MODEL TO cuda:{LOCAL_RANK}')
        model = model.to(f'cuda:{LOCAL_RANK}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def set_ff_model(_model, on_layers, embedding_requires_grad=True):
    mlp, ll, q_p, v_p, k_p, o_p = [], [], [], [], [], []

    for i, (name, param) in enumerate(_model.named_parameters()):
        param.requires_grad = False
        for ol in on_layers:
            if f"{ol}" in name:
                param.requires_grad = True
        if i == 0 and embedding_requires_grad:
            param.requires_grad = True

    for i, (name, param) in enumerate(_model.named_parameters()):
        print_rank_0("{:>5} : {:<60} : {:>15} => {:>25}".format(i, name, param.requires_grad, param.numel() / 1e6))
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
        for ol in on_layers:
            if f"{ol}" in name:
                ll.append(param.numel() / 1e6)

    print_rank_0(f'K Proj Contain {sum(k_p)} Million Parameters')
    print_rank_0(f'Q Proj Contain {sum(q_p)} Million Parameters')
    print_rank_0(f'O Proj Contain {sum(o_p)} Million Parameters')
    print_rank_0(f'V Proj Contain {sum(v_p)} Million Parameters')
    print_rank_0(f'MLP Contain    {sum(mlp)} Million Parameters')

    print_rank_0(f'\n------\nEach Block Contain {sum(ll)} Million Parameters (Based On {on_layers} Block)')

    train_able_parameters = 0
    for i, (name, param) in enumerate(_model.named_parameters()):
        train_able_parameters += param.numel() / 1e6 if param.requires_grad else 0
    print_rank_0(f'Total TrainAble Parameters In Model Is {train_able_parameters} Million Parameters')

    for i, (name, param) in enumerate(_model.named_parameters()):
        if param.requires_grad:
            print_rank_0("{:>5} : {:<60} : {:>15} => {:>25}".format(i, name, param.requires_grad, param.numel() / 1e6))
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


def main(on_layers,
         model_path,
         tokenizer_path,
         dataset_hub,
         path_to_save
         ):
    epochs = 2
    lr = 7e-5
    weight_decay = 1e-2
    torch_dtype = torch.float32
    embedding_requires_grad = True
    batch_size = 4

    model, tokenizer = configure_model(
        model_path,
        tokenizer_path,
        torch_dtype=torch_dtype
    )
    model = set_ff_model(model, on_layers=on_layers,
                         embedding_requires_grad=embedding_requires_grad)
    train_dataloader = configure_data_loader(dataset_hub, 'train', batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train(model, optimizer, train_dataloader, epochs=epochs)
    model.save_pretrained(path_to_save)


def transformer_trainer(on_layers,
                        model_path='erfanzar/PGT-3B',
                        tokenizer_path='erfanzar/PGT-3B',
                        per_gpu_train_batch_size: int = 8,
                        num_train_epochs=2,
                        learning_rate=1e-4,
                        embedding_requires_grad=True,
                        dataset_hub='erfanzar/Data-PGT-768',
                        output_dir='erfanzar/PGT-3B-T',
                        use_ddp=True
                        ):
    model, tokenizer = configure_model(model_path=model_path, tokenizer_path=tokenizer_path)
    model = set_ff_model(model, on_layers=on_layers, embedding_requires_grad=embedding_requires_grad)
    train_dataset = load_dataset(dataset_hub)

    extra = dict(
        fsdp='auto_wrap full_shard',
        fsdp_config={
            'fsdp_transformer_layer_cls_to_wrap': 'GPTNeoXLayer'
        },
    ) if not use_ddp else dict()
    args = TrainingArguments(
        per_gpu_train_batch_size=per_gpu_train_batch_size,
        weight_decay=0.01,
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        report_to='none',
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        fp16=True,
        do_eval=False,
        output_dir=output_dir,
        save_strategy='epoch',
        logging_dir=f"{output_dir}/logs",
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        **extra
    )
    trainer = Trainer(
        args=args,
        model=model,
        data_collator=DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer),
        train_dataset=train_dataset['train'],
        eval_dataset=None
    )
    trainer.train()


def accelerate_train(model_path='erfanzar/PGT-3B',
                     tokenizer_path='erfanzar/PGT-3B',
                     per_gpu_train_batch_size: int = 8,
                     num_train_epochs=2,
                     learning_rate=1e-4,
                     embedding_requires_grad=True,
                     dataset_hub='erfanzar/Data-PGT-768',
                     output_dir='erfanzar/PGT-3B-T',
                     gradient_accumulation_steps: int = 4):
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, tokenizer = configure_model(model_path=model_path, tokenizer_path=tokenizer_path, move_model=False)
    model = set_ff_model(model, [31, 30, 29, 28, 27, 26, 25, 24], embedding_requires_grad=embedding_requires_grad)
    dataloader = configure_data_loader(dataset_hub, field='train', batch_size=per_gpu_train_batch_size)
    # TODO


if __name__ == '__main__':
    transformer_trainer()
