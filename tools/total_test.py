from utils.utils import get_config_by_name

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
texts = """
PGT-S: 148.623552@
PGT-H: 329.589888@
PGT-X: 947.295744@
PGT-LX: 1917.48928@
PGT-LXX : 13297.537824@
LLama: 5243.834368@
LLmP-S: 148.81664@
LLmP-AL: 329.49248@
LLmP: 833.995776@
LLmP-X : 1567.580672@
LLmP-L: 1816.678208@
LLMoU-S : 148.136034@
LLMoU-ML : 329.708642@
LLMoU : 891.034722@
LLMoU-X : 1918.018658@
LLMoU-L : 2622.977122@
LLmPU-base : 598.639872@
LLmPU-S : 225.67808@
LLmPU-L : 758.303488@
LLmPU-LX : 1791.51744@
"""

if __name__ == "__main__":
    f = texts.split('@')
    print(f)
    # s = ({k, v} )
    # print(*s)
    s = {}

    for ka in [fa.split(':') for fa in f]:
        try:
            k, v = ka
            s[k.replace(' ', '').replace('\n', '')] = v.replace(' ', '').replace(r'\n', '')
        except ValueError:
            pass
    print(s)
    # hidden_size[0], num_layers[0], num_heads[0], max_sentence_length[0], params
    ska = '| Model | Hidden size | number of Layers | number of Heads | Max Sentence Length | Parameters |\n'
    cross = '|:------------|:------|:-------------------|------------|-----------|----------------|\n'
    ska += cross
    for model in models_name:

        # config = get_config_by_name(model, vocab_size=50274)
        # config.device = 'cpu'
        # if model.startswith('LLmPU'):
        #     m = LLmPUForConditionalGeneration(config)
        # elif model.startswith('LLMoU'):
        #     m = LLMoUModel(config)
        # elif model.startswith('LLmP'):
        #     m = LLmP(config)
        # elif model.startswith('LLama'):
        #     m = LLamaModel(config)
        # elif model.startswith('PGT'):
        #     m = PGT(config)
        # else:
        #     raise ValueError('Wrong Model ?')
        #
        # print(f'{model} : {count_model_parameters(m)}')
        # del m
        config = get_config_by_name(model, vocab_size=50274)
        config.device = 'cpu'
        if model.startswith('LLmPU'):

            hidden_size = config.d_model,
            num_layers = config.num_layers,
            num_heads = config.num_heads,
            max_sentence_length = config.max_length,
        elif model.startswith('LLMoU'):

            hidden_size = config.hidden_size,
            num_layers = config.n_layers,
            num_heads = config.n_heads,
            max_sentence_length = config.max_sentence_length,
        elif model.startswith('LLmP'):
            hidden_size = config.hidden_size,
            num_layers = config.n_layers,
            num_heads = config.n_heads,
            max_sentence_length = "ALiBi",
        elif model.startswith('LLama'):
            hidden_size = config.hidden_size,
            num_layers = config.n_layers,
            num_heads = config.n_heads,
            max_sentence_length = config.max_sentence_length,
        elif model.startswith('PGT'):
            hidden_size = config.hidden_size,
            num_layers = config.n_layers,
            num_heads = config.n_heads,
            max_sentence_length = config.max_sentence_length,
        else:
            raise ValueError('Wrong Model ?')
        try:
            params = "{:0,.2f}".format(float(s[model]))
            if len(params) > 6:
                params += ' B'
            else:
                params += ' M'
        except KeyError:
            params = ' > 15 B '
        args = [hidden_size[0], num_layers[0], num_heads[0], max_sentence_length[0], params]
        ska += f'| {model} | {" ".join(f"{f} | " for f in args)}\n'

    print(ska)
