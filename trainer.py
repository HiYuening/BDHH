from utils import *
from evaluation import evaluate_ranking
import random
import time
from model import Model
import torch.nn as nn

def training(dataLoader, config, device,n_node=None,n_price=None,n_category=None):
    if config.isTrain:
        numUsers = dataLoader.numTrain
        numItems = dataLoader.numItemsTrain
    else:
        numUsers = dataLoader.numTrainVal
        numItems = dataLoader.numItemsTest

    if numUsers % config.batch_size == 0:
        numBatch = numUsers // config.batch_size
    else:
        numBatch = numUsers // config.batch_size + 1


    idxList = [i for i in range(numUsers)]

    model = Model(config, numItems, adjacency=dataLoader.Train_adjacency,adjacency_pv=dataLoader.Train_adjacency_pv,adjacency_vp=dataLoader.Train_adjacency_vp,adjacency_pc=dataLoader.Train_adjacency_pc,adjacency_cp=dataLoader.Train_adjacency_cp,adjacency_cv=dataLoader.Train_adjacency_cv,adjacency_vc=dataLoader.Train_adjacency_vc,n_node=n_node,n_price=n_price,n_category = n_category,lr=config.lr, l2=config.l2, beta=config.beta, layers=config.layer,emb_size=config.emb_size, batch_size=config.batch_size,dataset=config.dataset, num_heads=config.num_heads,single_basket=dataLoader.single_basket,session_basket=dataLoader.session_basket).to(device)

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                        centered=False)

    torch.autograd.set_detect_anomaly(True)


    for epoch in range(config.epochs):
        # random.seed(1234)
        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batch_size * batch
            end = min(numUsers, start + config.batch_size)
            batchList = idxList[start:end]
            session_item, price_seqs, session_len, mask, reversed_sess_item, uHis, target,samples,sampleLen= generateBatchSamples(dataLoader, batchList, config, isEval=0,basket=False)
            session_item=torch.tensor(session_item).to(device)
            target = torch.from_numpy(target).type(torch.FloatTensor).to(device)
            scores = model.forward(session_item, price_seqs, session_len, mask, reversed_sess_item,samples,sampleLen)  

            loss = -(torch.log(scores) * target + torch.log(1 - scores) * (1 - target)).sum(-1).mean()
            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd = time.time()

        if epoch % config.evalEpoch == 0:
            timeEvalStar = time.time()
            print("start evaluation")

            recall, ndcg,mrr,hit,precision = evaluate_ranking(model, dataLoader, config, device, config.isTrain)
            timeEvalEnd = time.time()
            output_str = "Epoch %d \t recall@5=%.8f, recall@10=%.8f,recall@15=%.8f, recall@20=%.8f," \
                         "ndcg@5=%.8f, ndcg@10=%.8f, ndcg@15=%.8f, ndcg@20=%.8f,"\
                         "mrr@5=%.8f,mrr@10=%.8f,mrr@15=%.8f,mrr@20=%.8f," \
                          "hit@5=%.8f,hit@10=%.8f,hit@15=%.8f,hit@20=%.8f,"\
                          "precision@5=%.8f,precision@10=%.8f,precision@15=%.8f,precision@20=%.8f, [%.1f s]" % (
                             epoch + 1, recall[0], recall[1], recall[2], recall[3], ndcg[0], ndcg[1], ndcg[2],ndcg[3], mrr[0], mrr[1], mrr[2],mrr[3],hit[0], hit[1], hit[2],hit[3],precision[0], precision[1], precision[2],precision[3],
                             timeEvalEnd - timeEvalStar)
            print("time: %.1f, loss: %.3f" % (timeEpEnd - timeEpStr, epochLoss))
            print(output_str)
