from modules import RWKVConfig, RWKV_GPT_CasualLM, RWKVConfigTrain
from utils.utils import count_model_parameters

config = RWKVConfig(
    hidden_size=512,
    number_of_layers=8,
    vocab_size=32000,
)
train_config = RWKVConfigTrain()


def train():
    model = RWKV_GPT_CasualLM(config=config)
    print(f'Model Contain : {count_model_parameters(model)} M Parameters')


if __name__ == '__main__':
    train()
