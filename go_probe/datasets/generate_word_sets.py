from collections import defaultdict
import re

WORDS_PER_CATEGORY = 30

def count(words, annotations):
    freqs = defaultdict(lambda : 0)
    for ant in annotations:
        counted = set()
        cmt = ant['comments'].lower()
        for word in re.sub('[^A-Za-z0-9 ]+', '', cmt).split(' '):
            if len(word) < 2:
                continue
            if word in words and word not in counted:
                freqs[word] += 1
                counted.add(word)
    return freqs

def get_vocabulary(annotations):
    word_freqs = defaultdict(lambda : 0)
    for ant in annotations:
        cmt = ant['comments'].lower()
        for word in re.sub('[^A-Za-z0-9 ]+', '', cmt).split(' '):
            if len(word) < 2 or not word.isalpha():
                continue
            word_freqs[word] += 1
    return word_freqs

def main(fname, dict_file):
    import pickle as pkl
    ants = pkl.load(open(fname, 'rb'))

    print("Collecting annotation vocabulary")
    vocab = get_vocabulary(ants)
    print("Counting word frequencies")
    freqs = count(vocab, ants)
    word_freqs = [(w, freqs[w]) for w in vocab]
    word_freqs.sort(key=lambda x: -x[1])
    top_words = [x[0] for x in word_freqs[:WORDS_PER_CATEGORY]]
    print("Most frequent control words: ")
    print(top_words)

    with open(dict_file, 'r') as f:
        go_dict = f.readlines()
    go_dict = [w.strip() for w in go_dict]
    go_word_freqs = [(w, freqs[w]) for w in go_dict]
    go_word_freqs.sort(key=lambda x: -x[1])
    top_go_words = [x[0] for x in go_word_freqs[:WORDS_PER_CATEGORY]]
    print("Most freqeuent keywords: ")
    print(top_go_words)

    go_freq_set = [freqs[w] for w in top_go_words]
    min_freq, max_freq = min(go_freq_set), max(go_freq_set)
    similar_freq_words = [w for w in vocab if min_freq <= freqs[w] <= max_freq]
    
    import random
    selected_words = []
    random.seed(1)
    while len(selected_words) < WORDS_PER_CATEGORY:
        w = random.choice(similar_freq_words)
        if w not in top_go_words and w not in selected_words:
            selected_words.append(w)
    
    print("Similar frequency control words: ")
    print(selected_words)

    with open('top_control_words.txt', 'w') as f:
        for w in top_words:
            f.write(w + '\n')

    with open('sim_control_words.txt', 'w') as f:
        for w in selected_words:
            f.write(w + '\n')

    with open('top_go_words.txt', 'w') as f:
        for w in top_go_words:
            f.write(w + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='annotations_filtered.pkl')
    parser.add_argument('-g', '--go_dict', default='go_dict.txt')
    args = parser.parse_args()

    main(args.file, args.go_dict)