subjects = [
    'Artificial Intelligent',
    'Dubai',
    'United Arab Emirates',
    'Dubai buildings',
    'Abu Dhabi',
    'Saudi Arabia',
    'Burj Khalifa',
    'List of tallest buildings in Dubai',
    'Dubai Airports',
    'best hotels',
    'hotels in dubai'
]
# if __name__ == "__main__":
#     wikipedia.set_lang('en')
#     data = ''
#     for subject in subjects:
#         print(f'\033[1;36mCollecting {subject} Data ...')
#         data += '[SEP]'
#         data += wikipedia.summary(subject)
#         print(f'{subject} Data Collected *')
#     print(data)
#     with open('wiki.txt', 'w') as st:
#         st.write(data)

if __name__ == "__main__":

    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    url = 'https://www.funtrivia.com/en/Geography/United-Arab-Emirates-5810.html'
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
