from models import PTT
from datasets import load_dataset
from erutils.nlp import Lang
from cms import add_pad, add_special_tokens
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
if __name__ == "__main__":
    # Commenting model for first time
    vocab_size: int = tokenizer.vocab_size
    max_length: int = 512
    embedded: int = 512
    number_of_heads: int = 6
    number_of_layers: int = 8
    pad_index: int = tokenizer.pad_token_id
    # ptt = PTT(
    #     vocab_size=vocab_size,
    #     max_length=max_length,
    #     embedded=embedded,
    #     number_of_layers=number_of_layers,
    #     number_of_heads=number_of_heads,
    #     pad_index=pad_index
    # )
    squad_dataset = load_dataset('squad')
    train_data = squad_dataset['train']
    data_len = train_data.num_rows
    questions = train_data.data['question']
    answers = train_data.data['answers']
    # print(train_data)
    print("TOTAL TRAIN DATA : ", data_len)
    print('WORKED !!')
    print("EXAMPLE : ")
    print(f'VOCAB SIZE : {vocab_size}')
    q = str(questions[0])
    a = str(answers[0]['text'][0])
    print(q, " -> ", a)
    # tokenizer.add_tokens()
    qk = tokenizer.encode_plus(
        text=q,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    print(qk['input_ids'])
    print(' -> ')
    print(tokenizer.tokenize(str(a)))
