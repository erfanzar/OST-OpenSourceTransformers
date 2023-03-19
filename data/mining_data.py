import typing
from urllib.request import urlopen

from bs4 import BeautifulSoup
from wikipedia import wikipedia

BLACK_LIST = ['', 'Cart Epmty', 'HOME', 'MY Profile', 'MY Orders', 'MY Addresses', 'Search By', 'Gender ',
              'Lens Color ', 'Frame Material ', 'Frame Color ', 'Frame Shape ', 'Advance search', 'X',
              'Made ', 'Weight Group ', 'We Accept', 'Term & Conditions', 'Refund and Return Policy',
              'Privacy Policy', 'Contact Us', 'About Us', ' OPTICS4LESS ',
              'Welcome to Nice Optic Deira CityCenter',
              'Are you sure that you would like to proceed with the sample request ?', 'Brand ', 'Price',
              'Deira City Center, first floor, Opposite Carrefour', ' close '
              ]


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


BBK = [

    'ProductsShopsOpen your business0Cart Epmty0ProductsShopsxHOMEMY ProfileMY OrdersMY AddressesAboutContactusWelcome to Nice Optic Deira CityCenterAre you sure that you would like to proceed with the sample request ?SendCancelSunglassesEyeglassesContact LensSearch ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group Sort byLastestOldestAviator Retro X400% UV Protected, Polycarbonate & Polarized Lenses.\nEach Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Grudge*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 210Blame*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 200Casa Blanca*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 210Celebrity*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Cleopatra*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 220Deny*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Geometry*400% UV Protected & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Curious*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Debut*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Easy*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Emotion*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Girl*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Manpower X13*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Perspective X14*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Perspective X15*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 2001234Deira City Center, first floor, Opposite Carrefour close To Use The ChatBox We Need you Allow the Browser Notification  Click To Start Chat Your AccountYour AddressYour OrdersContact UsTerm & ConditionsPrivacy PolicyRefund and Return PolicyHomeAbout UsWe AcceptTerm & ConditionsRefund and Return PolicyPrivacy PolicyContact UsAbout UsCopyright Â© 2021-2022  OPTICS4LESS . All rights reserved.Design & Developed By',
    'ProductsShopsOpen your business0Cart Epmty0ProductsShops', 'ProductsShops', 'Open your business',
    '0Cart Epmty0ProductsShops', '0Cart Epmty', 'Cart Epmty', 'Cart Epmty', 'Cart Epmty', '0', '0', '0',
    'ProductsShops', 'ProductsShops', 'xHOMEMY ProfileMY OrdersMY AddressesAboutContactus', 'x', 'x',
    'HOME', 'MY Profile', 'MY Orders', 'MY Addresses',
    'Welcome to Nice Optic Deira CityCenterAre you sure that you would like to proceed with the sample request ?SendCancelSunglassesEyeglassesContact LensSearch ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group Sort byLastestOldestAviator Retro X400% UV Protected, Polycarbonate & Polarized Lenses.\nEach Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Grudge*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 210Blame*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 200Casa Blanca*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 210Celebrity*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Cleopatra*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 220Deny*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Geometry*400% UV Protected & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Curious*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Debut*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Easy*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Emotion*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Girl*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Manpower X13*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Perspective X14*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Perspective X15*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 2001234Deira City Center, first floor, Opposite Carrefour close To Use The ChatBox We Need you Allow the Browser Notification  Click To Start Chat ',
    'Welcome to Nice Optic Deira CityCenterAre you sure that you would like to proceed with the sample request ?SendCancelSunglassesEyeglassesContact LensSearch ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group Sort byLastestOldestAviator Retro X400% UV Protected, Polycarbonate & Polarized Lenses.\nEach Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Grudge*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 210Blame*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 200Casa Blanca*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 210Celebrity*400% UV Protected, Polycarbonate & Polarized Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Cleopatra*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 220Deny*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Geometry*400% UV Protected & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Curious*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Debut*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Easy*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Emotion*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Girl*400% UV Protected, Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.1AED 200Manpower X13*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Perspective X14*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.2AED 200Perspective X15*400% UV Protected, Polarized & Polycarbonate Lenses.\n*Each Pair of Vintage Sunglasses and Eyeglasses comes with 1 year warranty on manufacturing defects.3AED 2001234Deira City Center, first floor, Opposite Carrefour close To Use The ChatBox We Need you Allow the Browser Notification  Click To Start Chat ',
    'Welcome to Nice Optic Deira CityCenter',
    'Are you sure that you would like to proceed with the sample request ?SendCancel',
    'Are you sure that you would like to proceed with the sample request ?SendCancel', 'SendCancel',
    'SunglassesEyeglassesContact LensSearch ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group Sort byLastestOldest',
    'SunglassesEyeglassesContact LensSearch ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group Sort byLastestOldest',
    'SunglassesEyeglassesContact Lens', 'SunglassesEyeglassesContact Lens', 'SunglassesEyeglassesContact Lens',
    'Sunglasses', 'Eyeglasses', 'Contact Lens',
    'Search ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group Sort byLastestOldest',
    'Search ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group ',
    'Search ByBrand Gender Lens Color Frame Material Frame Color Frame Shape PriceAdvance searchXMade Weight Group ',
    'Search ByBrand Gender Lens Color Frame Material Frame Color Frame Shape Price',
    'Search ByBrand Gender Lens Color Frame Material Frame Color Frame Shape Price', 'Brand ', 'Gender ',
    'Lens Color ', 'Frame Material ', 'Frame Color ', 'Frame Shape ', 'Price',
    'Advance searchXMade Weight Group ', 'Advance searchXMade Weight Group ', 'Advance searchXMade Weight Group ',
    'Advance searchX', 'Made Weight Group ', 'Made Weight Group ', 'Made ', 'Weight Group ',
    'Sort byLastestOldest', '']


def collect_product(url='https://optics4less.com/brand/10/INVU/'):
    print('COLLECTING HTML')
    # html = urlopen(url).read()
    html = open(r'D:\Python\Document-Reader-CV\optics4less _ online platform.html', 'r')
    print('HTML COLLECTED')
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    print('EXTRACTED SCRIPTS')

    names = [c.text for c in soup.find_all('h3')]
    extra = [c.text for c in soup.find_all('span') if
             c.text not in BLACK_LIST]
    # print(dir(soup.find_all('div')[0].__dict__))

    _d = []
    for c in soup.find_all('div'):
        try:

            if c['class'] == ['text-xs', 'sm:flex', 'hidden', 'h-[100px]', 'overflow-y-auto', 'scrollbar-hide']:
                _d.append(c.text)
        except KeyError:
            pass
    _p = []
    _c = []
    # print(extra)
    for i in range(0, len(extra), 2):
        _p.append(extra[i + 1])
        _c.append(extra[i])

    string = ''
    for name, detail, colors, price in zip(names, _d, _c, _p):
        string += f' {name} is a sunglass and the detail of the {name} is [{detail}]' \
                  f' and available in {colors} different colors and the price of {name} is {price}'

    # print(string.replace('\n', '').replace('*',
    #                                        '') + f'there are totaly {len(names)} Glasses exist and the list of glasses are [{"".join(c for c in names)}]')
    print('-'.join(c for c in names))


if __name__ == "__main__":
    collect_product()
