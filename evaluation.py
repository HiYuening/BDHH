import torch
import numpy as np
import math
from utils import generateBatchSamples


def evaluate_ranking(model, dataLoader, config, device, isTrain):
    evalBatchSize = config.batch_size

    if isTrain:
        numUser = dataLoader.numValid
        numItems = dataLoader.numItemsTrain
    else:
        numUser = dataLoader.numTest
        numItems = dataLoader.numItemsTest

    if numUser % config.batch_size == 0:
        numBatch = numUser // evalBatchSize
    else:
        numBatch = numUser // evalBatchSize + 1

    idxList = [i for i in range(numUser)]

    Recall = []
    NDCG = []
    MRR=[]
    Hit=[]
    Precision=[]

    for batch in range(numBatch):
        start = batch * evalBatchSize
        end = min(batch * evalBatchSize + evalBatchSize, numUser)

        batchList = idxList[start:end]

        # target is the same with targetList in evaluation
        session_item, price_seqs, session_len, mask, reversed_sess_item, uHis, target,samples,sampleLen = generateBatchSamples(dataLoader,


        with torch.no_grad():
            scores = model.forward(session_item, price_seqs, session_len, mask, reversed_sess_item,samples,sampleLen)
            #scores = model.forward(samples, uHis, device)

        # get the index of top 40 items
        predIdx = torch.topk(scores, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList = target
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            targetList += target

    for k in [5, 10, 15, 20]:
        recall = calRecall(targetList, predIdxArray, k)
        Recall.append(recall)
        NDCG.append(calNDCG(targetList, predIdxArray, k))
        MRR.append(calMRR(targetList, predIdxArray, k))
        Hit.append(calHitRate(targetList, predIdxArray, k))
        Precision.append(calPrecision(targetList, predIdxArray, k))
    return Recall, NDCG,MRR,Hit,Precision


def calRecall(target, pred, k):
    assert len(target) == len(pred)
    sumRecall = 0
    for i in range(len(target)):
        gt = set(target[i])
        ptar = set(pred[i][:k])

        if len(gt) == 0:
            print('Error')

        sumRecall += len(gt & ptar) / float(len(gt))

    recall = sumRecall / float(len(target))

    return recall


def calNDCG(target, pred, k):
    assert len(target) == len(pred)
    sumNDCG = 0
    for i in range(len(target)):
        valK = min(k, len(target[i]))
        gt = set(target[i])
        idcg = calIDCG(valK)
        dcg = sum([int(pred[i][j] in gt) / math.log(j + 2, 2) for j in range(k)])
        sumNDCG += dcg / idcg

    return sumNDCG / float(len(target))


def calIDCG(k):
    return sum([1.0 / math.log(i + 2, 2) for i in range(k)])

def calPrecision(target, pred, k):
    assert len(target) == len(pred)
    sum_precision = 0.0
    for i in range(len(target)):
        gt_set = set(target[i])
        ptar = set(pred[i][:k])
        if len(ptar) > 0:
            sum_precision += len(gt_set & ptar) / float(len(ptar))
        else:
            sum_precision += 0
    return sum_precision / float(len(target))

def calHitRate(target, pred, k):
    assert len(target) == len(pred)
    hit_count = 0
    for i in range(len(target)):
        gt_set = set(target[i])
        if set(pred[i][:k]) & gt_set:
            hit_count += 1
    return hit_count / float(len(target))

def calMRR(target, pred, k):
    assert len(target) == len(pred)
    sum_rr = 0.0
    for i in range(len(target)):
        gt_set = set(target[i])
        for j, p in enumerate(pred[i][:k]):
            if p in gt_set:
                sum_rr += 1.0 / (j + 1)
                break
    return sum_rr / float(len(target))

