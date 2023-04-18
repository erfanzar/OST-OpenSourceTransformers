from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, Trainer
import torch
import transformers
from datasets import load_dataset
import evaluate
import numpy as np

model_id = 'erfanzar/LGeM-130M'

# metric = evaluate.load('f1')
#
#
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return metric.compute(predictions=predictions, references=labels, average="weighted")


dataset = load_dataset('json', data_files='/home/erfan/PycharmProjects/OST-OpenSourceTransformers/data/data-1k.jsonl')
tokenizer = LlamaTokenizer.from_pretrained('erfanzar/LGeM-7B')
config = LlamaConfig(
    vocab_size=len(tokenizer.get_vocab()),
    hidden_act='silu',
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=16,
    intermediate_size=1024 * 3,
    torchscript=False,
    torch_dtype=torch.float16,
    use_cache=False
)

# model = LlamaForCausalLM(config=config)
model = LlamaForCausalLM.from_pretrained('erfanzar/LGeM-130M')
tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset = dataset.map(
    lambda data_point: tokenizer(data_point['prompt'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))

args = transformers.TrainingArguments(
    # torch_compile=True,
    optim='adamw_torch_fused',
    # fp16=True,
    weight_decay=0.02,
    learning_rate=3e-4,
    output_dir=model_id,
    num_train_epochs=5,
    logging_dir=f'{model_id}/logs',
    logging_steps=50,
    logging_strategy='steps',
    save_strategy='epoch',
    report_to=['tensorboard'],
    save_total_limit=2,
    lr_scheduler_type='constant',
    auto_find_batch_size=True,
    # torch_compile_backend='inductor'
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    args=args,
    # callbacks=transformers.TrainerCallback()
)
# trainer.create_scheduler(800)

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=True)
