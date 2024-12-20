# -*- encoding: utf-8 -*-
'''
@Func    :   get word-level corpus
@Time    :   2021/03/05 16:47:38
@Author  :   Yixiao Ma 
@Contact :   mayx20@mails.tsinghua.edu.cn
'''

import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
# import thulac
import jieba
from deepthulac import LacModel
from sys import path
# path.append("/work/mayixiao/www22")
# from pre_ajjbqk import process_ajjbqk

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--d', type=str, default='/home/gaocheng/LeCaRD/data/documents/documents', help='Document dir path.')
parser.add_argument('--dpath', type=str, default='/home/gaocheng/LeCaRD/data/documents/document_path_short.json', help='Document_path file path.')
parser.add_argument('--s', type=str, default='/home/gaocheng/LeCaRD/data/others/stopword.txt', help='Stopword path.')
parser.add_argument('--m', type=str, default='jieba', help='The method of segmentation.')
parser.add_argument('--w', type=str, default='data/others/corpus_jieba.json', help='Write path.')

args = parser.parse_args()
if args.m == 'deepthulac':
    lac_seg = LacModel.load(path="/home/gaocheng/deepthulac/deepthulac-seg", device='cuda:0') # 加载模型，path为模型文件夹路径，SEG_MODEL表示自动从huggingface下载，device设置为cuda/cpu/mps
    lac_pos = LacModel.load(path="/home/gaocheng/deepthulac/deepthulac-pos", device='cuda:0') # 加载模型，path为模型文件夹路径，POS_MODEL表示自动从huggingface下载，device设置为cuda或cpu

def vocab(method, file_):
    if method == 'jieba':
        a = jieba.cut(file_['ajjbqk'], cut_all=False)
        tem = " ".join(a).split()
        return tem
    raise ValueError('Wrong method!')
# seg = thulac.thulac(seg_only=True, filt=True)

with open(args.dpath, 'r') as f:
    jspath = json.load(f)

with open(args.s, 'r') as g:
    lines = g.readlines()
stopwords = [i.strip() for i in lines]
stopwords.extend(['.','（','）','-'])

corpus = []
index_ridx = []

if args.m == "jieba":
    for path in tqdm(jspath['single'][:]):
        fullpath = os.path.join(args.d, path)
        with open(fullpath, 'r') as g:
            file_ = json.load(g)
        # if 'ajjbqk' in file_:
        # processed_file = process_ajjbqk(file_['ajjbqk'])
        # a = jieba.cut(processed_file, cut_all=False)
        # print(fullpath)
        if 'ajjbqk' not in file_:
            print(fullpath)
            continue
        tem = vocab(args.m, file_)
        # tem = seg.cut(file_['ajjbqk'], text = True).split()
        corpus.append([i for i in tem if not i in stopwords])

    for path0 in tqdm(jspath['retrial'][:]):
        for path in path0:
            fullpath = os.path.join(args.d, path)
            with open(fullpath, 'r') as g:
                file_ = json.load(g)
            # if 'ajjbqk' in file_:
            tem = vocab(args.m, file_)
            # tem = seg.cut(file_['ajjbqk'], text = True).split()
            corpus.append([i for i in tem if not i in stopwords])
elif args.m == "deepthulac":
    ajjbqk_list = []
    for path in jspath['single'][:]:
        fullpath = os.path.join(args.d, path)
        with open(fullpath, 'r') as g:
            file_ = json.load(g)
        if 'ajjbqk' not in file_:
            print(fullpath)
            continue
        ajjbqk_list.append(file_['ajjbqk'])
    for path0 in jspath['retrial'][:]:
        for path in path0:
            fullpath = os.path.join(args.d, path)
            with open(fullpath, 'r') as g:
                file_ = json.load(g)
            if 'ajjbqk' not in file_:
                print(fullpath)
                continue
            ajjbqk_list.append(file_['ajjbqk'])
    corpus = lac_seg.seg(ajjbqk_list, show_progress_bar=True)['seg']['res']

print(len(corpus))
with open(args.w, 'w') as f:
    json.dump(corpus, f, ensure_ascii=False)

