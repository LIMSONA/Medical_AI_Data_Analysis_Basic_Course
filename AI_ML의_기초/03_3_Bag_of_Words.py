# 실습: Bag of Words

import re

special_chars_remover = re.compile("[^\w'|_]")

def main():
    sentence = input()
    bow = create_BOW(sentence)

    print(bow)


def create_BOW(sentence):
    bow = {}
    sentence = sentence.lower()
    sentence = remove_special_characters(sentence)
    sentence = sentence.split()
    sentence = [i for i in sentence if len(i)>=1]
    for i in sentence:
        bow.setdefault(i,0)
    for j in sentence:
        bow[j]+=1
    
    return bow


def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


if __name__ == "__main__":
    main()