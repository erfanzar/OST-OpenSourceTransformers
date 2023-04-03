import logging
import math
import os
from typing import Optional, Union, Any, List

import accelerate
import erutils
import torch.distributed as dist
import torch.utils.data
from erutils.loggers import fprint
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config.config import TQDM_KWARGS
from modules import *
from modules.datasets import DatasetLLama, DatasetLGeM, DatasetLLmPChat, DatasetPGTChat, DatasetLLMoU, DatasetLLmP, \
    DatasetLLmPU, ManualDataSet
from utils.utils import save_checkpoints, get_config_by_name, device_info, get_memory, count_model_parameters, \
    create_output_path, compile_model, accelerate_mode, make2d, prompt_to_instruction

logger = logging.getLogger(__name__)

MODELS_CLASSES = Union[
    LGeMModel,
    LGeMForCausalLM,
    torch.nn.Module,
    LLamaModel,
    LLMoUModel,
    LLmPUForConditionalGeneration,
    LLmPUModel,
    PGT,
    LLmP,
    Any
]
CONFIG_CLASSES = Union[
    PGTConfig,
    LLmPConfig,
    LLmPUConfig,
    LGeMConfig,
    LLMoUConfig,
    LLamaConfig,
]

DATASET_CLASSES = Union[
    DatasetLLama,
    DatasetLGeM,
    DatasetLLmPChat,
    DatasetPGTChat,
    DatasetLLMoU,
    DatasetLLmP,
    DatasetLLmPU,
    ManualDataSet
]


def loss_cal(logits, label, *arg):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def load_from_weights(
        _weight: Union[str, os.PathLike],
        _model_class: MODELS_CLASSES,
        _accelerate: accelerate.Accelerator
):
    """

    :param _weight: path to weight
    :param _model_class: model_class to init model
    :param _accelerate: the accelerator
    :return: Tuple[model, optimizer, configuration, start_epoch, at]
    """
    erutils.fprint('Loading Checkpoints up on previous run ... ')
    checkpoints = torch.load(_weight, 'cpu')
    _configuration = checkpoints['configuration']

    erutils.loggers.show_hyper_parameters(_configuration)
    with _accelerate.init_empty_weights():
        model = _model_class(
            config=_configuration)
    optimizer_kwargs = dict(lr=_configuration.lr, weight_decay=_configuration.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    model_parameters_size: Optional[float] = count_model_parameters(model)
    model.load_state_dict(checkpoints['model'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoints['optimizer'])
    start_epoch = checkpoints['epoch']
    at = checkpoints['at']
    del checkpoints
    fprint(f'Model Loaded With {model_parameters_size} Million Parameters up on previous checkpoints')
    return model, optimizer, _configuration, start_epoch, at


def train(model_type,
          model_class: MODELS_CLASSES,
          weight,
          gradient_accumulation_steps,
          out_path: Union[str, os.PathLike],
          batch_size,
          dataset,
          use_compile=True,
          do_train=True,
          load_on_weights: Optional[bool] = False,
          question: Optional[str] = None,
          device_ids: List[int] = None,
          use_ddp: Optional[bool] = False,
          use_dp: Optional[bool] = False,
          world_size: int = 1,
          backend: Optional[str] = "gloo",
          rank: Optional[int] = 0,
          save_on_step: Optional[int] = 5000,
          init_method: Optional[str] = 'tcp://127.0.0.1:80',
          use_jit: bool = True,
          auto: bool = True):
    os.environ['USE_JIT'] = '1' if use_jit else '0'

    if gradient_accumulation_steps > 2 and auto:
        logger.info(f'gradient_accumulation_steps is higher than 2 setting it back to 1 for better performance')
        gradient_accumulation_steps = 1
    if load_on_weights:
        assert weight is not None, 'load_on_weight is used that mean model will build ' \
                                   'upon the previous weight you can\'t pass weight to be None '
    if use_ddp and use_dp:
        raise ValueError('You can\'t set both use ddp and use dp options at same time '
                         f'\n\t use_dpp : {use_ddp}'
                         f'\n\t use_dp : {use_dp}')
    color: str = '\033[1;94m'
    erutils.fprint(f'USING DistributedDataParallel : {use_ddp}', color=color)
    erutils.fprint(
        f' INFO DistributedDataParallel '
        f'\n\tbackend : {backend}'
        f'\n\tinit_method : {init_method}'
        f'\n\trank : {rank}'
        f'\n\tworld_size : {world_size}',
        color=color)
    if use_ddp:

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '80'
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    else:

        erutils.fprint(f' IGNORE DistributedDataParallel ...', color=color)
    assert hasattr(dataset, 'tokenizer'), 'dataset must contain tokenizer'
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    device = accelerator.device
    if weight is None:
        out_path = create_output_path(path=out_path, name=model_type)
        if not os.path.exists(os.path.join(str(out_path), 'weights')):
            os.mkdir(os.path.join(out_path, 'weights'))
    else:
        if weight.endswith('.pt'):
            out_path = weight.split('/')
            if 'weights' in out_path:
                out_path = os.path.join(*(p for p in out_path[:-2]))
            else:
                out_path = os.path.join(*(p for p in out_path[:-1]))
        else:
            raise ValueError('weight must contain path to .pt file')
    device_info()
    if load_on_weights:
        model, optimizer, configuration, start_epoch, at = load_from_weights(_weight=weight, _model_class=model_class,
                                                                             _accelerate=accelerator)
    else:
        configuration: CONFIG_CLASSES = get_config_by_name(model_type)

        configuration.vocab_size = dataset.tokenizer.vocab_size
        configuration.device = device
        configuration.batch_size = batch_size

        erutils.loggers.show_hyper_parameters(configuration)

        fprint('Loading Model ...' if weight else 'Creating Model ...')

        if weight is None:
            model = model_class(config=configuration).to(device)
        else:
            with accelerate.init_empty_weights():
                model = model_class(
                    config=configuration)

        optimizer_kwargs = dict(lr=configuration.lr, weight_decay=configuration.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        model_parameters_size: Optional[float] = count_model_parameters(model)

        checkpoints = torch.load(weight, 'cpu') if weight is not None else None

        if checkpoints is not None:
            try:
                model = accelerate.load_checkpoint_in_model(model, checkpoints['model'], device_map='auto',
                                                            offload_folder='offload', dtype=torch.float16)
                # model.load_state_dict()
                model = model.to(device)
                optimizer.load_state_dict(checkpoints['optimizer'])
                start_epoch = checkpoints['epoch']
                at = checkpoints['at']
                del checkpoints
            except Exception as err:
                print(f'checkpoint Loading failed make sure that you using right ckpt s : [ERROR] {err}')
                start_epoch = 0
                at = 0
        else:
            start_epoch = 0
            at = 0

        fprint(
            f'Model Loaded With {model_parameters_size} Million Parameters' if weight is not None
            else f'Model Created With {model_parameters_size} Million Parameters')

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configuration.batch_size, num_workers=4,
                                             pin_memory=True)
    if use_jit:
        model = torch.jit.script(model)
    if use_compile:
        model = compile_model(model)

    board = SummaryWriter(log_dir=f'{out_path}/tensorboard', filename_suffix=f'{model_type}') if do_train else None
    model = model.to(device=device)
    question = prompt_to_instruction(question) if question is not None else None
    q_ = dataset.pre_processing(question) if question is not None else None
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    if use_dp:
        model = torch.nn.parallel.DataParallel(model, device_ids=device_ids
                                               )
    model, optimizer, dataloader = accelerate_mode(accelerator=accelerator, model=model, optimizer=optimizer,
                                                   dataloader=dataloader)

    if do_train:
        logger.info('TRAIN IS ABOUT TO START')
        for epoch in range(start_epoch, configuration.epochs):
            loss_avg = 0

            with tqdm(enumerate(dataloader), **TQDM_KWARGS,
                      total=math.ceil(dataset.__len__() // configuration.batch_size)) as progress_bar:
                for i, (input_ids, attention_mask) in progress_bar:
                    logger.debug(f'\033[1;94m input_ids_t    : {input_ids.shape}')
                    logger.debug(f'\033[1;94m attention_mask : {attention_mask.shape}')
                    with accelerator.accumulate(model):
                        input_ids: Optional[torch.Tensor] = make2d(input_ids.type(torch.long).to(device))
                        logger.debug('RUNNING TRAIN FUNCTION IN MAIN THREAD ')

                        _, loss = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)

                        loss_avg += loss.item()

                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    free_gpu, used_gpu, total_gpu = get_memory(0)
                    if ((i + 1) % 50) == 0:
                        if question is not None and not use_jit:
                            tk = q_['input_ids']
                            tk = tk.to(configuration.device)
                            cals = []
                            try:
                                for pred in model.generate(tokens=tk, pad_id=dataset.tokenizer.pad_token_id,
                                                           attention_mask=None,
                                                           eos_id=dataset.tokenizer.eos_token_id):
                                    cals.append(pred)
                                cals = torch.cat(cals, dim=-1)
                                cals = cals.to('cpu')
                                awn = dataset.tokenizer.decode(cals[0])
                            except Exception as err:
                                awn = f'EMPTY {err}'
                            del cals
                            board.add_text('train/Context', f'{question}', global_step=at)
                            board.add_text('train/GeneratedResponse', f'{awn}', global_step=at)
                        board.add_scalar('train/Loss', scalar_value=loss.item(), global_step=at)
                        board.add_scalar('train/avg-Loss', scalar_value=(loss_avg / (i + 1)),
                                         global_step=at)

                    at += 1
                    progress_bar.set_postfix(epoch=f'[{epoch}/{configuration.epochs}]', device=configuration.device,
                                             loss_avg=(loss_avg / (i + 1)),
                                             loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)
                    if ((i + 1) % save_on_step) == 0:
                        save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         epochs=configuration.epochs, at=at,
                                         configuration=configuration,
                                         epoch=epoch + 1, config=model_type,
                                         name=f'{out_path}/weights/{model_type}-model.pt')
                print()
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epochs=configuration.epochs, at=at,
                                 configuration=configuration,
                                 epoch=epoch + 1, config=model_type,
                                 name=f'{out_path}/weights/{model_type}-model.pt')
                progress_bar.write('==> MODEL SAVED SUCCESSFULLY')

    else:
        with tqdm(range(1), **TQDM_KWARGS,
                  total=1) as progress_bar:
            for i in progress_bar:
                (input_ids, attention_mask) = dataset.__getitem__(i)
                logger.debug(f'\033[1;94m input_ids_t    : {input_ids.shape}')
                logger.debug(f'\033[1;94m attention_mask : {attention_mask.shape}')

                with accelerator.accumulate(model):
                    input_ids: Optional[torch.Tensor] = make2d(input_ids.type(torch.long).to(device))
                    logger.debug('RUNNING TRAIN FUNCTION IN MAIN THREAD ')

                    _, loss = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                free_gpu, used_gpu, total_gpu = get_memory(0)

                progress_bar.set_postfix(device=configuration.device,

                                         loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)

    dist.destroy_process_group()
