import logging
import os

os.chdir('/'.join(os.getcwd().split('\\')[:-1]))
from core.LLmPU_loading import load_llmpu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)
if __name__ == "__main__":
    model, tokenizer = load_llmpu(r'E:\Checkpoints\LLmPU-base\LLmPU-base-config.json',
                                  r'E:\Checkpoints\LLmPU-base\LLmPUForConditionalGeneration.pt',
                                  'google/flan-t5-base')
    while True:
        tok = tokenizer.encode_plus('Please answer the following question. What is the boiling point of Nitrogen?',
                                    max_length=512, pad_to_max_length=True,
                                    truncation=True, padding="max_length", return_tensors='pt')

        prediction = model.generate(tok['input_ids'])
        print(tokenizer.decode(prediction[0], skip_special_tokens=True))
