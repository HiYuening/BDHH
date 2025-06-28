import numpy as np
import torch


def generateBatchSamples(dataLoader, batchIdx, config, isEval,basket=False):
    if basket==False:
        samples, sampleLen, uHis, target = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)#sampleLen:每个user中篮子内的项目数(这里没用)

        all_itemSeq=[]
        all_priceSeq = []
        for sample in samples:
            single_itemSeq=[]
            single_priceSeq = []
            for basket in sample[:-1]:
                for seq in basket:
                    single_itemSeq.append(seq[0])
                    single_priceSeq.append(seq[1])
            all_itemSeq.append(single_itemSeq)
            all_priceSeq.append(single_priceSeq)


        num_node=[]
        for user in all_itemSeq:
            num_node.append(len(np.nonzero(user)[0]))  # calculate interacted items of each user
        max_n_node = np.max(num_node)

        session_item=[] # interacted items of each user
        price_seqs = [] # coresponding price of each item
        session_len = [] # the number of items of each uer
        reversed_sess_item = []
        mask = []
        for session, price in zip(all_itemSeq, all_priceSeq):
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            session_item.append(session + (max_n_node - len(nonzero_elems)) * [0])
            price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return session_item,price_seqs,session_len,mask,reversed_sess_item,uHis,target,[s[:-1] for s in samples],[s[:-1] for s in sampleLen]
