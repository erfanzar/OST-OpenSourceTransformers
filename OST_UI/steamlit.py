import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, logging
import torch
import textwrap

import datetime
from dataclasses import field, dataclass
from transformers import HfArgumentParser

import whisper
from streamlit_chat import message

logger = logging.get_logger(__name__)


@dataclass
class LoadConfig:
    mode: str = field(default='gui-chat', metadata={'help': 'mode to use ai in '})
    model_id: str = field(default='erfanzar/GT-J', metadata={'help': 'model to load'})
    load_model: bool = field(default=False, metadata={'help': "load model set to false for debug mode"})
    torch_type: torch.dtype = field(default=torch.float16, metadata={'help': "data type"})
    load_in_8bit: bool = field(default=False,
                               metadata={
                                   'help': "load model in 8 bit to make the models smaller "
                                           "and faster but its not recommended 😀 "})
    whisper_model: str = field(default='base', metadata={'help': 'model to load for whisper '})
    use_custom: bool = field(default=False, metadata={
        'help': 'use pipeline or custom generate func'
    })
    use_lgem_stoper: bool = field(default=False)


@st.cache_resource
def load_model(config: LoadConfig):
    logger.info(f'Loading model FROM : {config.model_id}')
    _model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_type,
        trust_remote_code=True,
        device_map='auto'
    ) if config.load_model else None
    model_whisper = whisper.load_model(config.whisper_model)
    logger.info(
        f'Done Loading Model with {(sum(m.numel() for m in _model.parameters()) / 1e9) if _model is not None else "NONE"} Billion Parameters')
    logger.info(f'Loading Tokenizer FROM : {config.model_id}')
    _tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.pad_token_id = _tokenizer.eos_token_id
    logger.info('Done Loading Tokenizer')
    return _model, _tokenizer, model_whisper


def generate(model: AutoModelForCausalLM, tokenizer, text: str, max_stream_tokens: int = 1,
             use_prompt_to_instruction: bool = False, generation_config=None, max_length=1536,
             b_pair=False):
    text = prompt_to_instruction(text) if use_prompt_to_instruction else text

    for i in range(max_stream_tokens):
        enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        text_r = text
        enc = model.generate(enc.input_ids.to(model.device),
                             attention_mask=enc.attention_mask.to(model.device),
                             generation_config=generation_config,
                             )
        text = tokenizer.decode(enc[0], skip_special_tokens=False)
        text = text[:-4] + tokenizer.eos_token if text[-4:] == '\n\n\n\n' else text
        lan_ = len('<|endoftext|>')
        text = text[:lan_] + tokenizer.eos_token if text[lan_:] == '<|endoftext|>' else text
        text = remove_spaces_between_tokens(text, '</s>', '<|ai|>')
        text = remove_spaces_between_tokens(text, '</s>', '<|prompter|>')
        text = remove_spaces_between_tokens(text, '<|prompter|>', '</s>')
        if text.endswith(tokenizer.eos_token) or text.endswith('\n\n\n\n') or text.endswith('<|endoftext|>'):
            yield text[len(text_r):] if b_pair else text
            break
        else:
            yield text[len(text_r):] if b_pair else text


def verify_text(txt):
    return '\n'.join([textwrap.fill(txt, width=110) for txt in txt.split('\n')])


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
            if '<|endoftext|>' in byte[len(text):]:
                break
            yield byte[len(text) - 1:].replace('<|endoftext|>', '')

        answer = final_res[len(text):len(final_res) - len('<|endoftext|>')]
        cache.append([original_text, answer])
        return '', cache


def prompt_to_instruction(text: str):
    return f"<|prompter|> {text} <|endoftext|><|assistant|>"


def prompt_to_instruction_lgem(text: str):
    return f"<|prompter|> {text} </s><|ai|>:"


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
        opt = f'<|prompter|> today is {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} make sure' \
              f' to stay polite and smart </s><|ai|>: OK ! </s>'
    else:
        opt = ''
        for f in cache_:
            opt += f"<|prompter|>{f[0]}</s><|ai|>:{f[1]}</s>"

    return opt


def remove_spaces_between_tokens(text, token1, token2):
    import re
    pattern = re.compile(re.escape(token1) + r'\s+' + re.escape(token2))
    return pattern.sub(token1 + token2, text)


def chat_bot_run(text: str,
                 cache,
                 max_steam_tokens,
                 max_new_tokens,
                 max_length,
                 temperature,
                 top_p,
                 top_k,
                 repetition_penalty,
                 voice, use_cache):
    if voice is not None:
        text_rec = whisper_model.transcribe(voice)['text']
        if text == '':
            text = text_rec
    if config_.use_lgem_stoper:
        opt = sort_cache_lgem(cache)
        text = opt + prompt_to_instruction_lgem(text)
    else:
        opt = sort_cache_pgt(cache)
        text = opt + prompt_to_instruction(text)
    final_res = ''
    generation_config = GenerationConfig(
        max_length=max_length,
        max_new_tokens=max_steam_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=use_cache
    )

    if config_.use_lgem_stoper:
        if model is not None:
            for byte in generate(model, tokenizer, text=text, b_pair=False,
                                 generation_config=generation_config, max_stream_tokens=max_new_tokens,
                                 max_length=max_length,
                                 use_prompt_to_instruction=False):
                final_res = byte
                chosen_byte = byte[len(text):]

                yield chosen_byte
            answer = final_res[len(text):len(final_res)]
        else:
            answer = 'It seems like im down or im not loaded yet 😇'
    else:
        if model is not None:
            for byte in generate(model, tokenizer, text=text, b_pair=False,
                                 generation_config=generation_config, max_stream_tokens=max_new_tokens,
                                 max_length=max_length,
                                 use_prompt_to_instruction=False):
                final_res = byte
                chosen_byte = byte[len(text):].replace('<|endoftext|>', '')
                yield chosen_byte
            answer = final_res[len(text):len(final_res) - len('<|endoftext|>')]
        else:
            answer = 'It seems like im down or im not loaded yet 😇'
    return '', answer


if __name__ == "__main__":
    config_: LoadConfig = HfArgumentParser(LoadConfig).parse_args_into_dataclasses()[0]
    if config_.load_model:
        model, tokenizer, whisper_model = load_model(config=config_)

        model = model.cuda() if model is not None else model
        whisper_model = whisper_model.cuda() if whisper_model is not None else whisper_model
        mcc = Conversation(model=model, tokenizer=tokenizer, config=config_)

    with st.form('input_bar', True):
        c1, c2 = st.columns([4, 1])
        text = c1.text_input(
            label='Text Input',
            placeholder="and you said ...",
            label_visibility="collapsed",
        )
        submit = c2.form_submit_button('Submit')

    max_new_tokens = st.sidebar.slider('Max New Tokens', max_value=2048, min_value=1, value=2048, step=1)
    max_length = st.sidebar.slider('Max Length', max_value=2048, min_value=256, value=2048, step=1)
    max_stream_tokens = st.sidebar.slider('Max Stream Tokens', value=1, max_value=8, min_value=1, step=1)
    temperature = st.sidebar.slider('Temperature', min_value=0.1, max_value=1.0, value=0.95, step=0.01)
    top_p = st.sidebar.slider('Top P', max_value=0.99, min_value=0.1, value=0.95, step=0.01)
    top_k = st.sidebar.slider('Top K', max_value=100, min_value=1, value=50, step=1)
    penalty = st.sidebar.slider('Repetition Penalty', max_value=5.0, min_value=1.0, value=1.2, step=0.1)
    stop = st.sidebar.button('Stop')
    clear = st.sidebar.button('Clear Conversation')
    use_cache = st.sidebar.checkbox('Use Cache', value=True)

    st.title(
        f'LucidBrains {config_.model_id}'
    )
