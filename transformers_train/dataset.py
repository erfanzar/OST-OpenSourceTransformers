from datasets import load_dataset
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('erfanzar/LGeM-7B')

tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset_train_non_cc = load_dataset('json',
                                    data_files='data/alpaca_data.json',
                                    split='train'
                                    )
dataset_train_cc = load_dataset('json',
                                data_files='/data/oasst_custom_valid_train.jsonl',
                                split='train',
                                field='train'
                                )

openassistant_oasst1 = load_dataset('h2oai/openassistant_oasst1')

dataset_eval = load_dataset('json',
                            data_files='data/oasst_custom_valid_train.jsonl',
                            field='validation', split='train')

openassistant_oasst1 = openassistant_oasst1.map(lambda x: {
    'edited': x['input'].replace('<human>:', 'User:').replace('<bot>:', '<|endoftext|>Assistant:') + '<|endoftext|>'})

openassistant_oasst1 = openassistant_oasst1.map(
    lambda data_point: tokenizer(data_point['edited'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))

dataset_train_cc = dataset_train_cc.map(
    lambda data_point: tokenizer(data_point['prompt'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))


def prompt_to_instruction(instruction, input_=None, response_=None, eos=tokenizer.eos_token):
    return f'User:{instruction} {input_}{eos}Assistant:{response_}{eos}'


def generate_prompt(data_point):
    ot = prompt_to_instruction(data_point['instruction'], data_point['input'], data_point['output'])
    return ot


dataset_train_non_cc = dataset_train_non_cc.map(
    lambda dp: {'prompt': generate_prompt(dp)}
)
dataset_train_non_cc = dataset_train_non_cc.map(
    lambda data_point: tokenizer(data_point['prompt'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False)
)

dataset_eval = dataset_eval.map(
    lambda data_point: tokenizer(data_point['prompt'], max_length=512, padding='max_length',
                                 truncation=True,
                                 add_special_tokens=False))
