import accelerate
import transformers
from IPython.core.display_functions import clear_output
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, logging
import torch
import textwrap
import os
from dataclasses import field, dataclass
from transformers import HfArgumentParser
import gradio as gr

try:
    import whisper
except:
    pass
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

logger = logging.get_logger(__name__)


class Seafoam(Base):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.emerald,
            secondary_hue: colors.Color | str = colors.blue,
            neutral_hue: colors.Color | str = colors.gray,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_lg,
            font: fonts.Font | str
            = (
                    fonts.GoogleFont("Quicksand"),
                    "ui-sans-serif",
                    "sans-serif",
            ),
            font_mono: fonts.Font | str
            = (
                    fonts.GoogleFont("IBM Plex Mono"),
                    "ui-monospace",
                    "monospace",
            ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,

        )
        super().set(
            body_background_fill="linear-gradient(80deg, *secondary_900, *neutral_800)",
            body_background_fill_dark="linear-gradient(80deg, *secondary_900, *neutral_800)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="0px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="4px",
        )


seafoam = Seafoam()


# logging.set_verbosity_info()


@dataclass
class LoadConfig:
    mode: str = field(default='gui-chat', metadata={'help': 'mode to use ai in '})
    model_id: str = field(default='erfanzar/GT-J', metadata={'help': 'model to load'})
    load_model: bool = field(default=True, metadata={'help': "load model set to false for debug mode"})
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
    theme_id: str = field(default='none')
    use_land: bool = field(default=False)
    block_name: str = field(default='none')
    use_n_eos: bool = field(default=False)


def load_model(config: LoadConfig):
    logger.info(f'Loading model FROM : {config.model_id}')
    if not config.use_land:
        _model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            load_in_8bit=config.load_in_8bit,
            torch_dtype=config.torch_type,
            trust_remote_code=True,
            device_map='auto'
        ) if config.load_model else None
        clear_output()
    else:
        clear_output()
        print("""
            This Process will take longer time in order to use models that are not built by huggingface and in are not 
            available in transformers library this option will add extra required options to module_class(pytorch)
            so device map can be used so model will load 2 times first time model will load without any weight and biases
            just to initialize and next time will load model and use it
        """)
        with accelerate.init_empty_weights():

            assert config.block_name != 'none', 'if you are using land option to use auto map for devices you ' \
                                                'must pass block name for model for example ' \
                                                'mpt model block name is GPTBlock'
            # config = AutoConfig.from_pretrained(
            #     config.model_id,
            #     trust_remote_code=True
            # )
            _model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                trust_remote_code=True
            )
            model_class = type(_model)
            del _model
            model_class._no_split_modules = [config.block_name]

        _model = model_class.from_pretrained(config.model_id,
                                             load_in_8bit=config.load_in_8bit,
                                             torch_dtype=config.torch_type,
                                             trust_remote_code=True,
                                             device_map='auto',
                                             ) if config.load_model else None

    model_whisper = whisper.load_model(config.whisper_model) if config.load_model else None
    logger.info(
        f'Done Loading Model with {(sum(m.numel() for m in _model.parameters()) / 1e9) if _model is not None else "NONE"}'
        f' Billion Parameters')
    logger.info(f'Loading Tokenizer FROM : {config.model_id}')
    _tokenizer = AutoTokenizer.from_pretrained(config.model_id,
                                               trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.pad_token_id = _tokenizer.eos_token_id
    logger.info('Done Loading Tokenizer')
    return _model, _tokenizer, model_whisper


def prompt_to_instruction(text: str):
    return f"{tokenizer.eos_token}<|prompter|> {text} {tokenizer.eos_token}<|assistant|>"


def prompt_to_instruction_n_eos(text: str):
    return f"<|prompter|> {text} <|assistant|>"


def prompt_to_instruction_lgem(text: str):
    return f"<|prompter|> {text} </s><|ai|>"


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

        # if config_.use_lgem_stoper:
        text = remove_spaces_between_tokens(text, '</s>', '<|assistant|>')
        text = remove_spaces_between_tokens(text, '</s>', '<|prompter|>')
        # text = remove_spaces_between_tokens(text, '<|prompter|>', '</s>')

        # text = text[:-4] + tokenizer.eos_token if text[-4:] == '\n\n\n\n' else text
        # lan_ = len('<|endoftext|>')
        # text = text[:lan_] + tokenizer.eos_token if text[lan_:] == '<|endoftext|>' else text
        if text.endswith(tokenizer.eos_token):
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
        for text in generate(model, tokenizer, text=user, max_stream_tokens=max_new_tokens, b_pair=byte_pair,
                             use_prompt_to_instruction=False):
            os.system('clear')
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
        opt = sort_cache_asst(cache)
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


def sort_cache_asst(cache_):
    if len(cache_) == 0:
        opt = ''
    else:
        opt = ''
        for f in cache_:
            opt += f"<|prompter|>{f[0]}{tokenizer.eos_token}<|assistant|>{f[1]}{tokenizer.eos_token}"

    return opt


def sort_cache_n_eos(cache_):
    if len(cache_) == 0:
        opt = ''
    else:
        opt = ''
        for f in cache_:
            opt += f"<|prompter|>{f[0]}<|assistant|>{f[1]}"

    return opt


def sort_cache_lgem(cache_):
    if len(cache_) == 0:
        opt = ''
    else:
        opt = ''
        for f in cache_:
            opt += f"<|prompter|>{f[0]}</s><|ai|>{f[1]}</s>"

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
        original_text = text
        text = opt + prompt_to_instruction_lgem(text)
    elif config_.use_n_eos:
        opt = sort_cache_n_eos(cache)
        original_text = text
        text = opt + prompt_to_instruction_n_eos(text)
    else:
        opt = sort_cache_asst(cache)
        original_text = text
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

    cache_f = cache
    cache_f.append([original_text, ''])

    if model is not None:
        for byte in generate(model, tokenizer, text=text, b_pair=False,
                             generation_config=generation_config, max_stream_tokens=max_new_tokens,
                             max_length=max_length,
                             use_prompt_to_instruction=False):
            final_res = byte
            clear_output()
            print(byte)
            chosen_byte = byte[len(text):].replace(tokenizer.eos_token, '')
            cache_f[-1][1] = chosen_byte
            yield '', cache_f
        answer = final_res[len(text):len(final_res) - len(tokenizer.eos_token)]
    else:
        answer = 'It seems like im down or im not loaded yet 😇'
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
    # theme = gr.themes.Soft(
    #     primary_hue="cyan",
    #     secondary_hue="teal",
    #     neutral_hue=gr.themes.Color(c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db",
    #                                 c400="#9ca3af", c50="#f9fafb", c500="#6b7280",
    #                                 c600="#4b5563", c700="#374151", c800="#1f2937",
    #                                 c900="#47a9c2", c950="#0b0f19"),
    # )

    with gr.Blocks(
            theme=gr.themes.Soft.from_hub(config_.theme_id) if config_.theme_id != 'none' else Seafoam()) as block:
        with gr.Row():
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(value=2048, maximum=3072, minimum=1, label='Max New Tokens', step=1)
                max_length = gr.Slider(value=2048, maximum=4096, minimum=1, label='Max Length', step=1)
                max_steam_tokens = gr.Slider(value=2, maximum=100, minimum=1, label='Max Stream Tokens', step=1,
                                             visible=True)
                temperature = gr.Slider(value=0.2, maximum=1, minimum=0.1, label='Temperature', step=0.01)
                top_p = gr.Slider(value=0.95, maximum=0.9999, minimum=0.1, label='Top P', step=0.01)
                top_k = gr.Slider(value=50, maximum=100, minimum=1, label='Top K', step=1)
                penalty = gr.Slider(value=1.2, maximum=5, minimum=1, label='Repetition Penalty', step=0.1, visible=True)
                # TODO
                use_cache = gr.Checkbox(label='Use Cache', value=False)

                voice = gr.Audio(source='microphone', type="filepath", streaming=False, label='Smart Voice', )
                stop = gr.Button(value='Stop ')
                clear = gr.Button(value='Clear Conversation')

            with gr.Column(scale=4):
                cache = gr.Chatbot(elem_id=main_class_conversation.config.model_id,
                                   label=main_class_conversation.config.model_id).style(container=True,
                                                                                        height=740)
        with gr.Row():
            with gr.Column(scale=1):
                submit = gr.Button(variant="primary")
            with gr.Column(scale=4):
                text = gr.Textbox(show_label=False).style(container=False)
        inputs = [text, cache, max_steam_tokens, max_new_tokens, max_length, temperature, top_p,
                  top_k,
                  penalty, voice, use_cache]
        sub_event = submit.click(fn=chat_bot_run,
                                 inputs=inputs,
                                 outputs=[text, cache])

        def stop_():
            ...

        def clear_():
            return []

        clear.click(fn=clear_, outputs=[cache])
        txt_event = text.submit(fn=chat_bot_run,
                                inputs=inputs,
                                outputs=[text, cache])

        stop.click(fn=None, inputs=None, outputs=None, cancels=[txt_event, sub_event])
        gr.Markdown(
            'LucidBrains is a platform that makes AI accessible and easy to use for everyone. '
            'Our mission is to empower individuals and businesses '
            'without requiring a background in data science or anything '
            'with using Artificial Intelligence and Pushing Technology Beyond Limits'
            '\n[OST-OpenSourceTransformers](https://github.com/erfanzar/OST-OpenSourceTransformers) From LucidBrains 🧠\n'
        )
    block.queue().launch(debug=True, share=True, inline=True, show_tips=True, width='100%', show_error=True,
                         max_threads=os.cpu_count(), )


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
    config_: LoadConfig = HfArgumentParser(LoadConfig).parse_args_into_dataclasses()[0]
    # config_ = LoadConfig()

    print(f'Running WITH MODE : {config_.mode}')
    model, tokenizer, whisper_model = load_model(config=config_)

    model = model.cuda() if model is not None else model
    whisper_model = whisper_model.cuda() if whisper_model is not None else whisper_model

    main(config_)
