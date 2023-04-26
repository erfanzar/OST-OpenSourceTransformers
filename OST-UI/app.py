from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import textwrap
import os
from dataclasses import field, dataclass
from transformers.utils import logging
from transformers import HfArgumentParser
import gradio as gr

logger = logging.get_logger(__name__)


@dataclass
class LoadConfig:
    mode: str = field(default='cli', metadata={'help': 'mode to use ai in '})
    model_id: str = field(default='erfanzar/PGT-1B-2EP', metadata={'help': 'model to load'})
    load_model: bool = field(default=False, metadata={'help': "load model set to false for debug mode"})
    torch_type: torch.dtype = field(default=torch.float32, metadata={'help': "data type"})
    load_in_8bit: bool = field(default=True,
                               metadata={
                                   'help': "load model in 8 bit to make the models smaller "
                                           "and faster but its not recommended 😀 "})


def load_model(config: LoadConfig):
    logger.info(f'Loading model FROM : {config.model_id}')
    _model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_type,
        device_map='auto'
    ) if load_model else None
    logger.info(f'Done Loading Model with {sum(m.numel() for m in _model.parameters()) / 1e9} Billion Parameters')
    logger.info(f'Loading Tokenizer FROM : {config.model_id}')
    _tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    logger.info('Done Loading Tokenizer')
    return _model, _tokenizer


def prompt_to_instruction(text: str):
    return f"<|prompter|> {text} <|endoftext|><|assistant|>"


def generate(model, tokenizer, text: str, max_new_tokens: int = 1024, use_prompt_to_instruction: bool = False,
             b_pair=False):
    text = prompt_to_instruction(text) if use_prompt_to_instruction else text

    for i in range(max_new_tokens):
        enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        text_r = text
        enc = model.generate(enc.input_ids, max_new_tokens=1, pad_token_id=0)
        text = tokenizer.decode(enc[0], skip_special_tokens=False)
        text = text[:-4] + tokenizer.eos_token if text[-4:] == '\n\n\n\n' else text
        if text.endswith(tokenizer.eos_token) or text.endswith('\n\n\n\n'):
            yield text[len(text_r):] if b_pair else text
            break
        else:
            yield text[len(text_r):] if b_pair else text


def verify_text(txt):
    return '\n'.join([textwrap.fill(txt, width=110) for txt in txt.split('\n')])


def conversation(model, tokenizer, cache=None, max_new_tokens=512, byte_pair=False):
    cache = '' if cache is None else cache
    while True:
        user = cache + prompt_to_instruction(input('>>  '))
        last_a = 'NONE'
        for text in generate(model, tokenizer, text=user, max_new_tokens=max_new_tokens, b_pair=byte_pair,
                             use_prompt_to_instruction=False):
            os.system('clear')
            print(verify_text(text).
                  replace('<|prompter|>', 'User : ').
                  replace('<|endoftext|><|assistant|>', '\nAI :').
                  replace('<|endoftext|>', '\n'), end='')
            last_a = text
        cache += last_a[len(cache):]


class Conversation:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run(self, text, max_new_tokens=512, cache=None):
        cache = '' if cache is None else cache
        text = cache + prompt_to_instruction(text)
        for byte in generate(self.model, self.tokenizer, text=text, max_new_tokens=max_new_tokens, b_pair=True,
                             use_prompt_to_instruction=False):
            yield byte


def gradio_ui(model, tokenizer):
    main_class_conversation = Conversation(model=model, tokenizer=tokenizer)
    interface = gr.Interface()
    interface.launch()


def main(config):
    print(f'Running WITH MODE : {config.mode}')
    model, tokenizer = load_model(config=config)
    if config.mode == 'cli':
        conversation(model=model, tokenizer=tokenizer)
    if config.mode == 'gui':
        gradio_ui(model=model, tokenizer=tokenizer)
    else:
        raise ValueError(f'Unknown Mode For : {config.mode}')


if __name__ == "__main__":
    config_ = HfArgumentParser(LoadConfig).parse_args_into_dataclasses()[0]
    main(config_)
