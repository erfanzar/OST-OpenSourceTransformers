import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model.eval()
text = "my name is siri and"
indexed_tokens = tokenizer.encode(text)

tokens_tensor = torch.tensor([indexed_tokens])

tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')
if __name__ == '__main__':
    for i in range(500):
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = torch.argmax(predictions[0, -1, :])

        tokens_tensor = torch.cat([tokens_tensor, predicted_index.unsqueeze(0).unsqueeze(0)], dim=-1)

        predicted_text = tokenizer.decode(predicted_index.cpu(), skip_special_tokens=True)
        print(f'{predicted_text}', end='')
