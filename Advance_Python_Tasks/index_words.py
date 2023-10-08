#Sample Text
"""
The Persian League is the largest sport event dedicated to the deprived areas of Iran. 
The Persian League promotes peace and friendship. 
This video was captured by one of our heroes who wishes peace.
"""

sample_text = input("Enter yor Text : ").replace(",","").split(" ")

counter = 1
index_words = []

for word in sample_text:
    if counter != 1:
        if sample_text[counter-2] != "." and sample_text[counter-2][-1] != ".":
            if word[0].isupper():
                index_words.append((counter,word))
    counter += 1
    
if index_words == []:
    print(None)
else:
    for word in index_words:
        print(f"{word[0]} : {word[1].strip('.')}")