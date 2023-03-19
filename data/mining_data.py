import typing
from urllib.request import urlopen

from bs4 import BeautifulSoup
from wikipedia import wikipedia


def wiki_data(subject: typing.Union[str, list[str]], lang: str = 'en', sep: str = ' '):
    wikipedia.set_lang(lang)
    data = ''
    for sbj in subject:
        print(f'\033[1;36mCollecting {sbj} Data ...')
        data += sep
        data += wikipedia.summary(sbj)
        print(f'{sbj} Data Collected *')

    print(data)
    with open('wiki.txt', 'w') as st:
        st.write(data)


def html_data(url: str):
    print('COLLECTING HTML')
    html = urlopen(url).read()
    print('HTML COLLECTED')
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    print('EXTRACTED SCRIPTS')
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    print('GETTING CHUNKS')
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '[SEP]'.join(chunk for chunk in chunks if chunk)
    print(text)


def collect_product():
    ...


if __name__ == "__main__":
    collect_product()
