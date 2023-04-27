from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, logging
import torch
import textwrap
import os
from dataclasses import field, dataclass
from transformers import HfArgumentParser
import gradio as gr
from typing import List, Optional

logger = logging.get_logger(__name__)
logging.set_verbosity_info()


@dataclass
class LoadConfig:
    mode: str = field(default='gui-chat', metadata={'help': 'mode to use ai in '})
    model_id: str = field(default='erfanzar/PGT-1B-2EP', metadata={'help': 'model to load'})
    load_model: bool = field(default=False, metadata={'help': "load model set to false for debug mode"})
    torch_type: torch.dtype = field(default=torch.float32, metadata={'help': "data type"})
    load_in_8bit: bool = field(default=False,
                               metadata={
                                   'help': "load model in 8 bit to make the models smaller "
                                           "and faster but its not recommended 😀 "})


def load_model(config: LoadConfig):
    logger.info(f'Loading model FROM : {config.model_id}')
    _model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_type,
    ) if load_model else None
    logger.info(f'Done Loading Model with {sum(m.numel() for m in _model.parameters()) / 1e9} Billion Parameters')
    logger.info(f'Loading Tokenizer FROM : {config.model_id}')
    _tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    logger.info('Done Loading Tokenizer')
    return _model, _tokenizer


def prompt_to_instruction(text: str):
    return f"<|prompter|> {text} <|endoftext|><|assistant|>"


def generate(model: AutoModelForCausalLM, tokenizer, text: str, max_new_tokens: int = 1024,
             use_prompt_to_instruction: bool = False, generation_config=None,
             b_pair=False):
    text = prompt_to_instruction(text) if use_prompt_to_instruction else text

    for i in range(max_new_tokens):
        enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        text_r = text
        enc = model.generate(enc.input_ids, generation_config=generation_config)
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
    def __init__(self, model, tokenizer, config):
        self.model: AutoModelForCausalLM = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.config: LoadConfig = config

    def run(self, text,
            cache, max_length, temperature, top_p, top_k,
            repetition_penalty
            ):
        opt = sort_cache_pgt(cache)
        original_text = text
        text = opt + prompt_to_instruction(text)
        final_res = ''
        generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=1,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        for byte in generate(self.model, self.tokenizer, text=text, b_pair=False,
                             generation_config=generation_config,
                             use_prompt_to_instruction=False):
            final_res = byte
            yield byte[len(text):].replace('<|endoftext|>', '')
        answer = final_res[len(text):len(final_res) - len('<|endoftext|>')]
        cache.append([original_text, answer])
        return '', cache


def sort_cache_pgt(cache_):
    if len(cache_) == 0:
        opt = ''
    else:
        opt = ''
        for f in cache_:
            opt += f"<|prompter|>{f[0]}<|endoftext|><|assistant|>{f[1]}<|endoftext|>"

    return opt


def sort_cache_lgem(cache_):
    if len(cache_) == 0:
        opt = ''
    else:
        opt = ''
        for f in cache_:
            opt += f"User:{f[0]}\nAI:{f[1]}"

    return opt


def chat_bot_run(text: str, cache, max_new_tokens,
                 max_length,
                 temperature,
                 top_p,
                 top_k,
                 repetition_penalty):
    opt = sort_cache_pgt(cache)
    original_text = text
    text = opt + prompt_to_instruction(text)
    final_res = ''
    generation_config = GenerationConfig(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
    cache_f = cache
    cache_f.append([original_text, ''])
    for byte in generate(model, tokenizer, text=text, b_pair=False,
                         generation_config=generation_config,
                         use_prompt_to_instruction=False):
        final_res = byte
        chosen_byte = byte[len(text):].replace('<|endoftext|>', '')
        cache_f[-1][1] = chosen_byte
        yield '', cache_f
    answer = final_res[len(text):len(final_res) - len('<|endoftext|>')]
    cache.append([original_text, answer])
    return '', cache


def gradio_ui(main_class_conversation):
    interface = gr.Interface(fn=main_class_conversation.run, outputs='text',
                             inputs=[gr.inputs.Textbox(lines=10, placeholder='Im just a placeholder ignore me ... '),
                                     gr.inputs.Slider(default=1024, maximum=1024, minimum=1, label='Max Length'),
                                     gr.inputs.Slider(default=0.9, maximum=1, minimum=0.2, label='Temperature'),
                                     gr.inputs.Slider(default=0.95, maximum=0.9999, minimum=0.1, label='Top P'),
                                     gr.inputs.Slider(default=50, maximum=100, minimum=1, label='Top K'),
                                     gr.inputs.Slider(default=1.2, maximum=5, minimum=1,
                                                      label='Repetition Penalty')])
    interface.queue()
    interface.launch(share=True)


def gradio_ui_chat(main_class_conversation: Conversation):
    with gr.Blocks(theme=gr.themes.Soft()) as block:
        gr.Markdown(
            f"""
            # {main_class_conversation.config.model_id} Is here To Assist You 
            \n\n## [OST-OpenSourceTransformers](https://github.com/erfanzar/OST-OpenSourceTransformers) From LucidBrains 🧠
            LucidBrains is a platform that makes AI accessible and easy to use for everyone.
            Our mission is to empower individuals and businesses
            with the tools they need to harness the power of AI and machine learning,
            without requiring a background in data science or anything we
            will just build what you want for you and help you to have better time and living life
             with using Artificial Intelligence and Pushing Technology Beyond Limits
            """)
        with gr.Row():
            with gr.Column(scale=4):
                cache = gr.Chatbot(elem_id=main_class_conversation.config.model_id,
                                   label=main_class_conversation.config.model_id).style(container=True,
                                                                                        height=680)
            with gr.Column(scale=1):
                max_length = gr.Slider(value=1024, maximum=1024, minimum=1, label='Max Length', step=1)
                max_steam_tokens = gr.Slider(value=1, maximum=3, minimum=1, label='Max Stream Tokens', step=1)
                temperature = gr.Slider(value=0.9, maximum=1, minimum=0.2, label='Temperature', step=0.01)
                top_p = gr.Slider(value=0.95, maximum=0.9999, minimum=0.1, label='Top P', step=0.01)
                top_k = gr.Slider(value=50, maximum=100, minimum=1, label='Top K', step=1)
                penalty = gr.Slider(value=1.2, maximum=5, minimum=1, label='Repetition Penalty', step=0.1, visible=True)
                # TODO
                penalty_ = gr.Slider(value=1.2, maximum=10, minimum=1, label='Repetition', step=0.1, visible=True)
                gre_mode = gr.Checkbox(label='Greedy Mode')
                smart_mode = gr.Checkbox(label='Smart Mode')
                informational_mode = gr.Checkbox(label='Informational Mode')
        with gr.Row():
            with gr.Column(scale=4):
                text = gr.Textbox(show_label=False).style(container=False)

            with gr.Column(scale=1):
                submit = gr.Button()

        submit.click(fn=chat_bot_run,
                     inputs=[text, cache, max_steam_tokens, max_length, temperature, top_p, top_k, penalty],
                     outputs=[text, cache])
        text.submit(fn=chat_bot_run,
                    inputs=[text, cache, max_steam_tokens, max_length, temperature, top_p, top_k, penalty],
                    outputs=[text, cache])

        block.queue().launch(debug=False, share=True)


def main(config):
    mcc = Conversation(model=model, tokenizer=tokenizer, config=config)
    if config.mode == 'cli':
        conversation(model=model, tokenizer=tokenizer)
    if config.mode == 'gui':
        gradio_ui(main_class_conversation=mcc)
    if config.mode == 'gui-chat':
        gradio_ui_chat(main_class_conversation=mcc)
    else:
        raise ValueError(f'Unknown Mode For : {config.mode}')


if __name__ == "__main__":
    config_ = HfArgumentParser(LoadConfig).parse_args_into_dataclasses()[0]
    print(f'Running WITH MODE : {config_.mode}')
    model, tokenizer = load_model(config=config_)
    main(config_)
