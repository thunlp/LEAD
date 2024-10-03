# -*- encoding: utf-8 -*-
'''
@Func    :   evaluation of retrieved results
@Time    :   2021/03/04 17:35:21
@Author  :   Yixiao Ma 
@Contact :   mayx20@mails.tsinghua.edu.cn
'''

import os
import numpy as np
import json
import math
import functools
import argparse
# from sklearn.metrics import ndcg_score
from tqdm import tqdm

def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    Pe  = float(ysum*xsum)/k**2
    P0 = float(P0/k*1.0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))
    return cohens_coefficient

def fleiss_kappa(testData, N, k, n): 
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0*dataMat[i, j]**2
        temp -= n
        temp /= (n-1)*n
        P0 += temp
    P0 = 1.0*P0/N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i]/sum)**2 # (1/k)**2
    Pe = ysum*oneMat*1.0
    ans = (P0-Pe)/(1-Pe)
    return ans[0, 0]

# def ndcg(ranks,K):
#     dcg_value = 0.
#     idcg_value = 0.
#     log_ki = []

#     sranks = sorted(ranks, reverse=True)

#     for i in range(0,K):
#         logi = math.log(i+2,2)
#         dcg_value += ranks[i] / logi
#         idcg_value += sranks[i] / logi

#     return dcg_value/idcg_value

def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    return dcg_value/idcg_value

def load_file(args):
    label_dic = json.load(open(args.label, 'r'))
    dic = json.load(open(args.pred, 'r'))
    # for key in list(label_dic.keys())[:]:
    #     dic[key].reverse()

    
    return label_dic, [dic]#, lawformer_dic

if __name__ == "__main__":
    METHOD = "easy"
    parser = argparse.ArgumentParser(description="Help info:")
    parser.add_argument('--m', type=str, choices= ['NDCG', 'P', 'MAP', 'KAPPA'], default='all', help='Metric.')
    # TODO path need to be changed
    parser.add_argument('--label', type=str, default='/yeesuanAI06/thunlp/gaocheng/LeCaRD/data/label/label_top30_dict.json', help='Label file path.')
    parser.add_argument('--pred', type=str, default="/yeesuanAI06/thunlp/gaocheng/bge-M3-results_LeCaRD.json", help='Prediction file path.')
    parser.add_argument('--q', type=str, choices= ['all', 'common', 'controversial', 'test', 'test_2'], default='all', help='query set')

    args = parser.parse_args()
    if ("CAIL2022" in args.pred or "cail" in args.pred) and "cail2022_train" not in args.pred:
        print("test on CAIL2022")
        # TODO path need to be changed
        args.label = "/yeesuanAI06/thunlp/gaocheng/CAIL2022/label/label_1.json"
    elif "DPR_rank_LeCard/" in args.pred:
        print("test on LeCaRD")
    label_dic, dics = load_file(args)
    if args.q == 'all':
        keys = list(label_dic.keys())
        print(f"total {len(keys)}")
    elif args.q == 'common':
        keys = list(label_dic.keys())[:77]  
    elif args.q == 'controversial':
        keys = list(label_dic.keys())[77:]
    elif args.q == 'test':
        keys = [i for i in list(label_dic.keys())[:100] if list(label_dic.keys()).index(i) % 5 == 0]
        # keys = [i for i in list(combdic.keys()) if list(combdic.keys()).index(i) % 5 == 0]
        # keys = label_dic.keys()
    elif args.q == 'test_2':
        keys = label_dic.keys()
        
    if args.m == 'P' or args.m == 'all': 
        topK_list = [5,10]
        sp_list = []

        for topK in topK_list:
            temK_list = []
            for dic in dics:
                sp = 0.0
                min = 10000
                mark_key = ""
                for key in keys:
                    x = label_dic[key]
                    if METHOD == "hard":
                        ranks = [i for i in dic[key]] 
                        sp += float(len([j for j in ranks[:topK] if (str(j) in label_dic[key] and label_dic[key][str(j)] == 3)])/topK)
                    else:
                        ranks = [i for i in dic[key] if str(i) in list(x.keys())] 
                        sp += float(len([j for j in ranks[:topK] if label_dic[key][str(j)] == 3])/topK)
                    # if fenshu < min:
                    #     min = fenshu
                    #     mark_key = key
                    # if key == "5156":
                    #     print(ranks)
                    #     print(sp)
                temK_list.append(sp/len(keys))
                # if topK == 5:
                #     print(f"前五个正确最少的是: {mark_key}, 分数为{min}")
            sp_list.append(f"P@{topK}: {temK_list}")
        print(sp_list)

    if args.m == 'MAP' or args.m == 'all':
        map_list = []
        for dic in dics:
            smap = 0.0
            for key in keys:
                if METHOD == "hard":
                    ranks = [i for i in dic[key]] #TODO
                    rels = [ranks.index(i) for i in ranks if (str(i) in label_dic[key] and label_dic[key][str(i)] == 3)]
                else:
                    ranks = [i for i in dic[key] if str(i) in label_dic[key]] 
                    rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)] == 3]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank+1] if (str(j) in label_dic[key] and label_dic[key][str(j)] == 3)])/(rel_rank+1))
                if len(rels) > 0:
                    smap += tem_map / len(rels)
            map_list.append(f"MAP: {smap/len(keys)}")
        print(map_list)

    if args.m == 'NDCG' or args.m == 'all':
        topK_list = [10, 20, 30]
        ndcg_list = []
        for topK in topK_list:
            temK_list = []
            for dic in dics:
                sndcg = 0.0
                for key in keys:
                    rawranks = []
                    for i in dic[key]:
                        if str(i) in list(label_dic[key]): # 原仓库取了前30个, 但这样应该是不对的, 有长度大于30的
                            rawranks.append(label_dic[key][str(i)])
                        else: # 换测评方式只要注释这两行
                            if METHOD == "hard":
                                rawranks.append(0)
                    ranks = rawranks + [0]*(len(list(label_dic[key]))-len(rawranks))
                    if sum(ranks) != 0:
                        _ndcg = ndcg(ranks, list(label_dic[key].values()), topK)
                        # print(f"{key}, {_ndcg}")
                        sndcg += _ndcg
                temK_list.append(sndcg/len(keys))
            ndcg_list.append(f"NDCG {topK}: {temK_list}")
        print(ndcg_list)
    
    if args.m == 'KAPPA':
        lists = json.load(open('/work/mayixiao/similar_case/LeCaRD/private/data/label_top30.json', 'r'))
        dataArr = []

        for i in lists[0].keys():
            for j in range(30):
                tem = [0,0,0,0]
                for k in range(3):
                    tem[int(lists[k][i][j])-1] += 1
                dataArr.append(tem)
        print(fleiss_kappa(dataArr, 30*len(lists[0]), 4, 3))

    # elif MODE == 'F1':
    #     topK = 15
    #     rdic_list = [tdic, ldic, bdic]
    #     f1_list = []
    #     for rdic in rdic_list:
    #         k = 0
    #         sf1 = 0.0
    #         for key in list(combdic.keys())[:100]:
    #             pre = 0.0
    #             recall = 0.0
    #             ranks = [i for i in rdic[key] if i in list(combdic[key][:30])] 
    #             pre = float(len([j for j in ranks[:topK] if label_dic[k][list(combdic[key][:30]).index(j)] == 1])/topK)
    #             allrel = len([j for j in ranks[:] if label_dic[k][list(combdic[key][:30]).index(j)] == 1])
    #             if allrel > 0 and pre > 0:
    #                 recall = float(len([j for j in ranks[:topK] if label_dic[k][list(combdic[key][:30]).index(j)] == 1])/allrel)
    #                 sf1 += 2/(1/pre+1/recall)
    #             k += 1
    #         f1_list.append(sf1/100)

    #     print(f1_list)
    