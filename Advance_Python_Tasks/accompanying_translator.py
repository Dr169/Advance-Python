number_of_word = int(input("Enter the number of words you want to translate : "))

english = {}
french = {}
german = {}

for word in range(number_of_word):
    word = input(f"Enter the word {word+1} : ").split(" ")
    english[word[1]] = word[0]
    french[word[2]] = word[0]
    german[word[3]] = word[0]

sentence = input("Enter your sentence : ").split(" ")

for index, word in enumerate(sentence):
    if word in english:
        sentence[index] = english[word]
    elif word in french:
        sentence[index] = french[word]
    elif word in german:
        sentence[index] = german[word]

print(' '.join(sentence))