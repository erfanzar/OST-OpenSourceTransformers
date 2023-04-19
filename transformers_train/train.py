from transformers import LlamaTokenizer, Trainer, LlamaForCausalLM, LlamaConfig
from modules import LGeMForCausalLM, LGeMConfig
import torch

import transformers
from datasets import load_dataset
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType

model_id = 'erfanzar/LGeM-100M'
dataset_train = load_dataset('json',
                             data_files='/home/erfan/PycharmProjects/OST-OpenSourceTransformers/data/oasst_custom_valid_train.jsonl',
                             field='train', split='train')

dataset_eval = load_dataset('json',
                            data_files='/home/erfan/PycharmProjects/OST-OpenSourceTransformers/data/oasst_custom_valid_train.jsonl',
                            field='validation', split='train')

tokenizer = LlamaTokenizer.from_pretrained('erfanzar/LGeM-7B')

tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# config = LGeMConfig(
#     vocab_size=len(tokenizer.get_vocab()),
#     hidden_size=768,
#     num_hidden_layers=14,
#     num_attention_heads=16,
#     intermediate_size=768 * 4,
#     bos_token_id=tokenizer.bos_token_id,
#     pad_token_id=tokenizer.pad_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=False
# )

config = LlamaConfig(
    vocab_size=len(tokenizer.get_vocab()),
    hidden_size=768,
    num_hidden_layers=10,
    num_attention_heads=12,
    intermediate_size=768 * 2,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False,
    torchscript=True, name_or_path=model_id,
    torch_dtype=torch.float16
)

# low_rank_config = LoraConfig(
#     r=config.hidden_size // config.num_attention_heads,
#     lora_alpha=32,
#     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'],
#     bias='none',
#     task_type=TaskType.CAUSAL_LM,
#     lora_dropout=0.05
# )

model = LlamaForCausalLM(config=config)
# model = LGeMForCausalLM(config=config)
# model = get_peft_model(prepare_model_for_int8_training(model), low_rank_config)
# model.print_trainable_parameters()
print(sum(m.numel() for m in model.parameters()) / 1e6, '  Million Parameters IN MODEL')
dataset_train = dataset_train.map(
    lambda data_point: tokenizer(data_point['prompt'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))

dataset_eval = dataset_eval.map(
    lambda data_point: tokenizer(data_point['prompt'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))

args = transformers.TrainingArguments(
    # torch_compile=True,
    optim='adamw_torch',
    # fp16=True,
    weight_decay=0.02,
    learning_rate=2e-4,
    output_dir=model_id,
    num_train_epochs=500,
    logging_dir=f'{model_id}/logs',
    logging_steps=50,
    logging_strategy='steps',
    save_strategy='steps',
    save_steps=500,
    report_to=['tensorboard', 'wandb'],
    save_total_limit=1,
    lr_scheduler_type='constant',
    auto_find_batch_size=True,
    evaluation_strategy='epoch',
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    args=args,
)

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=False)
