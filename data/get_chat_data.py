import erutils

# url = 'https://github.com/Pawandeep-prog/finetuned-gpt2-convai/blob/main/chat_data.json'
url = 'https://raw.githubusercontent.com/replicate/cog_stanford_alpaca/main/alpaca_data.json'
if __name__ == "__main__":
    erutils.download(url)
