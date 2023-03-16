from utils.utils import get_config_by_name, count_model_parameters
from modules import (LLmPUForConditionalGeneration,
                     LLmP,
                     LLmPUModel,
                     LLamaModel,
                     LLMoUModel,
                     PGT,
                     )

models_name = ['PGT-S',
               'PGT-M',
               'PGT-X',
               'PGT-LX',
               'PGT-LXX',
               'LLama',
               'LLmP-S',
               'LLmP-ML',
               'LLmP',
               'LLmP-X',
               'LLmP-L',
               'LLmP-LX',
               'LLMoU-S',
               'LLMoU-ML',
               'LLMoU',
               'LLMoU-X',
               'LLMoU-L',
               'LLMoU-LX',
               'LLmPU-base',
               'LLmPU-S',
               'LLmPU-L',
               'LLmPU-LX', ]

if __name__ == "__main__":
    for model in models_name:
        print(model)
        config = get_config_by_name(model)
        if model.startswith('LLmPU'):
            m = LLmPUForConditionalGeneration(config)
        elif model.startswith('LLMoU'):
            m = LLMoUModel(config)
        elif model.startswith('LLmP'):
            m = LLmP(config)
        elif model.startswith('LLama'):
            m = LLamaModel(config)
        elif model.startswith('PGT'):
            m = PGT(config)
        else:
            raise ValueError('Wrong Model ?')

        print(f'{model} : {count_model_parameters(m)}')
        del m
