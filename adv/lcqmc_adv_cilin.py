import thulac
import json
import sys
import os
import os.path as osp
from tqdm import tqdm
from word_similarity import WordSimilarity2010
ws_tool = WordSimilarity2010()
import numpy as np

thu = thulac.thulac()
word2sem = json.load(open('/home/anonymous/Sememe/data/word2sememe.json'))
word2id = json.load(open('/home/anonymous/Sememe/data/word2id.json'))
word2pos = json.load(open('/home/anonymous/Sememe/data/word2pos.json'))
word2topk = {}
correct1 = np.load('/home/anonymous/lcqmc.t.npy')
correct2 = np.load('/home/anonymous/lcqmc.tsep.npy')

fname = sys.argv[1]

def score(src_word, tgt_word):
    return ws_tool.similarity(src_word, tgt_word)

def get_top_k_words(src_word, label, top_k):
    if src_word not in word2id: return []
    if src_word in word2topk: return word2topk[src_word]
    word_with_score_list = []
    for tgt_word in word2sem:
        # ------------------------------------------------------------------------------
        # 1) Filter by vocab
        if tgt_word not in word2id: continue
        # 2) Filter by POS
        tgt_pos = word2pos.get(tgt_word, '')
        src_pos = {'n': 'noun', 'a': 'adj', 'd': 'adv'}[label]
        if src_pos not in tgt_pos: continue
        # ------------------------------------------------------------------------------
        word_with_score_list.append((tgt_word, score(src_word, tgt_word)))
    words, scores = zip(*sorted(word_with_score_list, key=lambda x:x[1], reverse=True))
    # print(words[:20], scores[:20])
    topk_words = []
    for word in words:
        if len(topk_words) >= top_k: break
        if word != src_word: topk_words.append(word)
    word2topk[src_word] = topk_words
    return topk_words

out_fnames = dict()
for pos in ['a', 'n', 'd']:
    if 'train' in fname:
        mode = 'train'
    elif 'test' in fname:
        mode = 'test'
    elif 'dev' in fname:
        mode = 'dev'
    out_fnames[pos] = fname.replace('_%s.tsv' % mode, '_advcilin_%s_%s.tsv' % (pos, mode))
print('Input:', fname)
print('Output:')
print(json.dumps(out_fnames, indent=4))

fouts = dict()
for pos, out_fname in out_fnames.items():
    fouts[pos] = open(out_fname, 'w')

with open(fname) as fin:
    for i, line in enumerate(tqdm(fin.readlines())):
        if correct1[i] == 0 or correct2[i] == 0: 
            print(i)
            continue
        splitline = line.strip().split('\t')
        text1, text2, text_label = splitline
        text1_cut = thu.cut(''.join(text1.replace('<unk>', '#').replace('<N>', '$').split()))
        text2_cut = thu.cut(''.join(text2.replace('<unk>', '#').replace('<N>', '$').split()))
        text1_split, label1 = zip(*text1_cut)
        text2_split, label2 = zip(*text2_cut)

        for i, (word, label) in enumerate(zip(text1_split, label1)):
            if label == 'n' or label == 'a' or label == 'd':
                similar_words = get_top_k_words(word, label, 3)
                for similar_word in similar_words:
                    new_text1_split = text1_split[:i] + (similar_word,) + text1_split[i+1:]
                    fouts[label].write(' '.join(new_text1_split + ('_start_',) + text2_split).replace('#', ' <unk> ').replace('$', ' <N> ').replace('  ', ' ') + '\t' + text_label + '\n')

        for i, (word, label) in enumerate(zip(text2_split, label2)):
            if label == 'n' or label == 'a' or label == 'd':
                similar_words = get_top_k_words(word, label, 3)
                for similar_word in similar_words:
                    new_text2_split = text2_split[:i] + (similar_word,) + text2_split[i+1:]
                    fouts[label].write(' '.join(text1_split + ('_start_',) + new_text2_split).replace('#', ' <unk> ').replace('$', ' <N> ').replace('  ', ' ') + '\t' + text_label + '\n')
                
for pos, fout in fouts.items():
    fout.close()        
        
        
        
