from gtts import gTTS
# from

import os

mytext = 'Mr ryaz do you have any problem with my existance ?'

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False, )
from playsound import playsound

# Saving the converted audio in a mp3 file named
# welcome
if __name__ == "__main__":
    myobj.save("welcome.mp3")
    playsound('E:\\Programming\\Python\\Ai-Projects\\MODEL P\\preprocessing train\\welcome.mp3')

# Playing the converted file
