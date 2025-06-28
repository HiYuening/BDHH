from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np
import pickle
import time
import csv



class dataLoader():
    def __init__(self, config):
        dataRoot = './data/' + config.dataset + '.pkl'
        #{'user'：[[('item','price','category','basket'),()],[(),()]]}
        with open(dataRoot, 'rb') as f:
            dataDict = pickle.load(f)

        num=0
        for i in dataDict.values():
            num=num+len(i)

        user2item = self.generate_user_list(dataDict)# useless
        count_remove,user2idx = self.generate_sets(user2item)
        assert count_remove==0

        # count the number of items, categories, prices, baskets, including training, validation, testing sets
        self.numItems,self.numPrice,self.numCategory,self.numBasket = self.get_num_item_price_category_basket()

        #print("num_users_removed   %d" % numRe)
        print("num_valid_users   %d" % len(self.testList))
        print("num_items   %d" % self.numItems)


        self.numTrain, self.numValid, self.numTrainVal, self.numTest = len(self.testList), len(self.testList), len(
            self.testList), len(self.testList)
        # same number of items for training and testing
        self.numItemsTrain, self.numItemsTest = self.numItems, self.numItems
        # same id in training, validation and testing
        self.valid2train = {}
        self.test2trainVal = {}
        for i in range(len(self.trainList)):
            self.valid2train[i] = i
            self.test2trainVal[i] = i

        if config.isTrain:
            self.lenTrain = self.generateLens(self.trainList)
            self.lenVal = self.generateLens(self.validList)
        else:  # Test
            self.lenTrainVal = self.generateLens(self.trainValList)
            self.lenTest = self.generateLens(self.testList)

        start = time.time()
        if config.isTrain:
            # generate and store the matrices in the first run and just load these matrices in the following runs
            self.uhisMatTra, self.tarMatTra,self.Train_adjacency,self.Train_adjacency_pv,self.Train_adjacency_vp,self.Train_adjacency_pc,self.Train_adjacency_cp,self.Train_adjacency_cv,self.Train_adjacency_vc,self.single_basket,self.session_basket = self.generateHis(self.trainList, isTrain=1, isEval=0)
            self.uhisMatVal, self.tarMatVal,self.Test_adjacency,self.Test_adjacency_pv,self.Test_adjacency_vp,self.Test_adjacency_pc,self.Test_adjacency_cp,self.Test_adjacency_cv,self.Test_adjacency_vc,self.single_basket,self.session_basket = self.generateHis(self.validList, isTrain=1, isEval=1)
            # check
            print(self.are_coo_matrices_equal(self.Train_adjacency, self.Test_adjacency)) #False
            print('Mat done')



        else:
            # generate and store the matrices in the first run and just load these matrices in the following runs
            self.uhisMatTraVal, self.tarMatTraVal,self.Train_adjacency,self.Train_adjacency_pv,self.Train_adjacency_vp,self.Train_adjacency_pc,self.Train_adjacency_cp,self.Train_adjacency_cv,self.Train_adjacency_vc,self.sgle_basket,self.session_basket = self.generateHis(self.trainValList, isTrain=0, isEval=0)
            self.uhisMatTest, self.tarMatTest,self.Test_adjacency,self.Test_adjacency_pv,self.Test_adjacency_vp,self.Test_adjacency_pc,self.Test_adjacency_cp,self.Test_adjacency_cv,self.Test_adjacency_vc,self.single_basket,self.session_basket = self.generateHis(self.testList, isTrain=0, isEval=1)
            # check
            print(self.are_coo_matrices_equal(self.Train_adjacency, self.Test_adjacency)) #False
            print('Mat done')

        print("finish generating his matrix, elaspe: %.3f" % (time.time() - start))

    # def generate_user_list(self, trainDict, validDict, testDict):
    def generate_user_list(self, dataDict):
        all_users = list(dataDict.keys())
        user2item = {}
        for user in all_users:
            user2item[user] = dataDict.get(user, [])
        return user2item#user2item==dataDict==True

    def are_coo_matrices_equal(self,mat1, mat2):
        # check if the shapes are same
        if mat1.shape != mat2.shape:
            return False

        # change to CSR
        mat1_csr = mat1.tocsr()
        mat2_csr = mat2.tocsr()


        return (np.array_equal(mat1_csr.indptr, mat2_csr.indptr) and
                np.array_equal(mat1_csr.indices, mat2_csr.indices) and
                np.array_equal(mat1_csr.data, mat2_csr.data))

    def generate_sets(self, user2item):
        self.trainList = []
        self.validList = []
        self.trainValList = []
        self.testList = []


        self.user2item=[]


        count = 0
        count_remove = 0
        user2idx = {}

        for user in user2item:
            # only keep the users with validation and testing baskets
            if len(user2item[user]) < 4:  # train>=2, valid=1, test=1
                count_remove += 1
                continue
            user2idx[user] = count
            count += 1
            self.trainList.append(user2item[user][:-2])
            self.validList.append(user2item[user][:-1])
            self.trainValList.append(user2item[user][:-1])
            self.testList.append(user2item[user])

            self.user2item.append(user2item[user][:])
        #return count_remove, user2idx
        return count_remove,user2idx

    def get_num_item_price_category_basket(self):
        numItem = 0
        numPrice = 0
        numCategory = 0
        numBasket = 0
        for baskets in self.testList:
            # all the baskets of users
            for basket in baskets:
                for seq in basket:
                    if isinstance(seq[0], int):
                        numItem = max(seq[0], numItem)
                        numPrice = max(seq[1], numPrice)
                        numCategory = max(seq[2], numCategory)
                        numBasket = max(seq[3], numBasket)
                    else:
                        numItem = max(int(seq[0]), numItem)
                        numPrice = max(int(seq[1]), numPrice)
                        numCategory = max(int(seq[2]), numCategory)
                        numBasket = max(int(seq[3]), numBasket)

        return numItem,numPrice,numCategory,numBasket

    def find_zeros(self, matrix):
        zero_positions = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    zero_positions.append((i, j))
        return zero_positions
    def generateHis(self, userList, isTrain, isEval):

        # pre generate the history vector and target vector for each user
        if isTrain and not isEval:
            hisMat= np.zeros((self.numTrain, self.numItemsTrain))
        elif isTrain and isEval:
            hisMat = np.zeros((self.numValid, self.numItemsTrain))
        elif not isTrain and not isEval:
            hisMat = np.zeros((self.numTrainVal, self.numItemsTest))
        else:
            hisMat = np.zeros((self.numTest, self.numItemsTest))

        if not isEval and isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTrain)) #(user number，item number)
        elif not isEval and not isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTest))
        else:
            tarMat = []

        his_user2item_Mat = np.zeros((self.numTrain, self.numItems))
        his_price2item_Mat = np.zeros((self.numPrice, self.numItems))
        his_price2category_Mat = np.zeros((self.numPrice, self.numCategory))
        his_category2item_Mat = np.zeros((self.numCategory, self.numItems))
        for i in range(self.numTrain):
            for Bas in userList[i][:-1]:
                for seq in Bas:
                    # ID starts from 1, using -1 in order to avoid overflowing
                    his_user2item_Mat[i, seq[0]-1] = 1

        # recording item, price, and category in each basket
        flag_single=0
        single_basket_item=[]
        single_basket_price = []
        single_basket_category = []

        session_basket_item=[]
        session_basket_price = []
        session_basket_category = []

        #训练集和测试集统一使用
        for i in range(self.numTrain):
            for Bas in self.user2item[i]:
                for seq in Bas:
                    his_price2item_Mat[seq[1] - 1, seq[0] - 1] = 1
                    his_price2category_Mat[seq[1]-1, seq[2]-1] = 1
                    his_category2item_Mat[seq[2]-1, seq[0]-1] = 1

        for i in range(len(userList)):
            trainUser = userList[i][:-1]
            targetUser = userList[i][-1]

            for Bas in trainUser:
                for seq in Bas:
                    hisMat[i, seq[0]-1] += 1      #useless
            if not isEval:
                for seq in targetUser:
                    tarMat[i, seq[0]-1] = 1
            else:
                #tarMat.append(targetUser)
                tarMat.append([seq[0]-1 for seq in targetUser])


        hisMat = csr_matrix(hisMat) #(num_user，num_item+1)--->didn't remove duplicates
        his_user2item_Mat = csr_matrix(his_user2item_Mat) #(num_user，num_item+1)--->matrics with only 0 or 1
        his_price2item_Mat = csr_matrix(his_price2item_Mat) #(num_price+1，num_item+1)--->matrics with only 0 or 1
        his_price2category_Mat = csr_matrix(his_price2category_Mat) #(num_price+1，num_category+1)--->matrics with only 0 or 1
        his_category2item_Mat = csr_matrix(his_category2item_Mat) #(num_category+1，num_item+1)--->matrics with only 0 or 1

        #T：useritem，p：price，v：item，c：category
        #adjacency_vp
        H_T=his_user2item_Mat
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        H_pv=his_price2item_Mat
        BH_pv = H_pv
        BH_vp = H_pv.T

        H_pc=his_price2category_Mat
        BH_pc = H_pc
        BH_cp = H_pc.T

        H_cv=his_category2item_Mat
        BH_cv = H_cv
        BH_vc = H_cv.T

        adjacency = DHBH_T.tocoo()
        adjacency_pv = BH_pv.tocoo()
        adjacency_vp = BH_vp.tocoo()
        adjacency_pc = BH_pc.tocoo()
        adjacency_cp = BH_cp.tocoo()
        adjacency_cv = BH_cv.tocoo()
        adjacency_vc = BH_vc.tocoo()



        if not isEval:
            tarMat = csr_matrix(tarMat)

        uHisMat = normalize(hisMat, norm='l1', axis=1, copy=False)

        single_basket=dict()
        single_basket['item']=single_basket_item
        single_basket['price'] = single_basket_price
        single_basket['category'] = single_basket_category

        session_basket=dict()
        session_basket['item']=session_basket_item
        session_basket['price'] = session_basket_price
        session_basket['category'] = session_basket_category
        return uHisMat, tarMat,adjacency,adjacency_pv,adjacency_vp,adjacency_pc,adjacency_cp,adjacency_cv,adjacency_vc,single_basket,session_basket

    def batchLoader(self, batchIdx, isTrain, isEval):
        if isTrain and not isEval:
            train = [self.trainList[idx] for idx in batchIdx]
            trainLen = [self.lenTrain[idx] for idx in batchIdx]
            uHis = self.uhisMatTra[batchIdx, :].todense()
            target = self.tarMatTra[batchIdx, :].todense()
        elif isTrain and isEval:
            train = [self.validList[idx] for idx in batchIdx]
            trainLen = [self.lenVal[idx] for idx in batchIdx]
            uHis = self.uhisMatVal[batchIdx, :].todense()
            target = [self.tarMatVal[idx] for idx in batchIdx]
        elif not isTrain and not isEval:
            train = [self.trainValList[idx] for idx in batchIdx]
            trainLen = [self.lenTrainVal[idx] for idx in batchIdx]
            uHis = self.uhisMatTraVal[batchIdx, :].todense()
            target = self.tarMatTraVal[batchIdx, :].todense()
        else:
            train = [self.testList[idx] for idx in batchIdx]
            trainLen = [self.lenTest[idx] for idx in batchIdx]
            uHis = self.uhisMatTest[batchIdx, :].todense()
            target = [self.tarMatTest[idx] for idx in batchIdx]

        return train, trainLen, uHis, target

    def generateLens(self, userList):
        # list of list of lens of baskets
        lens = []
        # pre-calculate the len of each sequence and basket
        for user in userList:
            lenUser = []
            # the last bas is the traget to calculate errors
            trainEUser = user[:-1]
            for bas in trainEUser:
                lenUser.append(len(bas))
            lens.append(lenUser)
        # the number of interacted baskets for each user, e.g., user0 bought 6 items in basket0, and 8 items in basket1, then lens[0]=[6,8]
        return lens