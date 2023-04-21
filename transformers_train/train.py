from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    LlamaConfig
from datasets import load_dataset
from peft import LoraConfig

model_id = 'LGeM-1B'

tokenizer = LlamaTokenizer.from_pretrained('erfanzar/LGeM-7B')

tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

config = LlamaConfig(
    vocab_size=len(tokenizer.get_vocab()),
    hidden_size=1536,
    num_hidden_layers=16,
    num_attention_heads=16,
    intermediate_size=1536 * 6,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False,
    torchscript=True,
    name_or_path=model_id,
)
model = LlamaForCausalLM(config=config)


def prompt_to_instruction(instruction, input_=None, response_=None, eos=tokenizer.eos_token):
    return f'User:{instruction} {input_}{eos}Assistant:{response_}{eos}'


def generate_prompt(data_point):
    ot = prompt_to_instruction(data_point['instruction'], data_point['input'], data_point['output'])
    return ot


openassistant_oasst1 = load_dataset('h2oai/openassistant_oasst1')

openassistant_oasst1 = openassistant_oasst1.map(lambda x: {
    'edited': x['input'].replace('<human>:', 'User:').replace('<bot>:', '<|endoftext|>Assistant:') + '<|endoftext|>'})

openassistant_oasst1 = openassistant_oasst1.map(
    lambda data_point: tokenizer(data_point['edited'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=openassistant_oasst1['train'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    args=TrainingArguments(
        # torch_compile=True,
        optim='adamw_torch',
        fp16=True,
        weight_decay=0.02,
        learning_rate=2e-4,
        output_dir=f"/content/drive/MyDrive/OST-OpenSourceTransformers/transformers_train/{model_id}",
        num_train_epochs=50,
        logging_dir=f'{model_id}/logs',
        logging_steps=1,
        save_strategy='steps',
        save_steps=500,
        report_to=['tensorboard', 'wandb'],
        save_total_limit=1,
        lr_scheduler_type='constant',
        auto_find_batch_size=True,
        evaluation_strategy='epoch',
        do_train=True,
        # gradient_accumulation_steps=4,
        gradient_checkpointing=True,

        push_to_hub_model_id=model_id
    ),
)

if __name__ == '__main__':
    trainer.train(resume_from_checkpoint=False)
