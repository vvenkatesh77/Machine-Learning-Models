#Tokenization into sentences and word


from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
#sentence tokenization _Splitting sentences in the paragraph

text="Hellow Venky.How are you.What are you doing now"
print(sent_tokenize(text))

#Word Tokenizing
from nltk.tokenize import word_tokenize

text="Hellow venkatesh.How are you,today weather is worm"
print(word_tokenize(text))



 

  
