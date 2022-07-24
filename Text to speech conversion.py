

from newspaper import Article
import nltk
from gtts import gTTS
import os
'''
test="hi how are you, what you are doiing right now, how was the weather outside"

#nltk.download('punkt')

language='en'
myobj=gTTS(text=test, lang=language, slow=False)

myobj.save("read_articleapr.mp3")
os.system("start read_articleapr.mp3")
'''

#article =Article('https://www.bbc.com/news/world-europe-62249015')
article =Article('https://www.bbc.com/hindi/international-62248451')


article.download()
article.parse()
#nltk.download('puntk')
article.nlp()
mytext=article.text
#language='en'
language='hi'
#language='tel'
myobj=gTTS(text=mytext,lang=language,slow=False)

myobj.save("read_articleapr2.mp3")
os.system("start read_articleapr2.mp3")









