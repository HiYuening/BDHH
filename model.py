import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch.nn import Module, Parameter
import numpy as np
import math
from all_attentions import SelfAttentionLayer

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class PowerBasket(Module):
    def __init__(self, emb_size):
        super(PowerBasket, self).__init__()
        self.emb_size = emb_size

        self.fc1 = nn.Linear(emb_size, emb_size)
        self.fc2 = nn.Linear(emb_size, emb_size)

        self.proj_item_1 = nn.Sequential(
            nn.Linear(emb_size, emb_size//2),
            nn.ReLU(),
            nn.Linear(emb_size//2, emb_size),
        )
        self.proj_item_2 = nn.Linear(emb_size, emb_size)


        self.proj_price_1 = nn.Sequential(
            nn.Linear(emb_size, emb_size//2),
            nn.ReLU(),
            nn.Linear(emb_size//2, emb_size),
        )
        self.proj_price_2 = nn.Linear(emb_size, emb_size)

        self.item_norm = nn.Linear(self.emb_size * 2, self.emb_size)
        self.category_norm = nn.Linear(self.emb_size * 2, self.emb_size)
        self.price_norm = nn.Linear(self.emb_size * 2, self.emb_size)
        self.item_gate_norm = nn.Linear(self.emb_size, 1)
        self.price_gate_norm = nn.Linear(self.emb_size, 1)


        self.self_attn_item = SelfAttentionLayer(self.emb_size, self.emb_size)
        self.self_attn_price = SelfAttentionLayer(self.emb_size, self.emb_size)
        self.self_attn_category = SelfAttentionLayer(self.emb_size, self.emb_size)
    def forward(self, item_embeddings, price_embeddings, category_embeddings,samples,sampleLen):
        # record item, price, and category in each basket
        flag_single=0
        single_basket_item=[]
        single_basket_price = []
        single_basket_category = []

        basket2item_price_category = dict()
        basket2item_price_category['item']=single_basket_item
        basket2item_price_category['price'] = single_basket_price
        basket2item_price_category['category']=single_basket_category
        len_basket=len(basket2item_price_category['item'])


        item2basket = dict()
        price2basket = dict()
        category2basket = dict()

        for b in range(len_basket):
            len_seqs=len(basket2item_price_category['item'][b])
            for s in range(len_seqs):
                if basket2item_price_category['item'][b][s] not in item2basket:
                    item2basket[basket2item_price_category['item'][b][s]] = []
                item2basket[basket2item_price_category['item'][b][s]].append(b)

                if basket2item_price_category['price'][b][s] not in price2basket:
                    price2basket[basket2item_price_category['price'][b][s]] = []
                price2basket[basket2item_price_category['price'][b][s]].append(b)

                if basket2item_price_category['category'][b][s] not in category2basket:
                    category2basket[basket2item_price_category['category'][b][s]] = []
                category2basket[basket2item_price_category['category'][b][s]].append(b)


        basket_n = len(basket2item_price_category["item"])

        basket_embedding_item = []
        basket_embedding_price = []
        basket_embedding_category = []
        for basket_id,_ in enumerate(basket2item_price_category["item"]):
            basket_embedding_item.append(torch.mean(item_embeddings[basket2item_price_category["item"][basket_id]],dim=0))
            basket_embedding_price.append(
                torch.mean(price_embeddings[basket2item_price_category["price"][basket_id]],dim=0))
            basket_embedding_category.append(
                torch.mean(category_embeddings[basket2item_price_category["category"][basket_id]],dim=0))

        basket_embedding_item = torch.tanh(torch.stack(basket_embedding_item))
        basket_embedding_price = torch.tanh(torch.stack(basket_embedding_price))


        #new_item_embeddings
        for item_id, basket_seq_id in item2basket.items():
            if len(basket_seq_id)<2:
                    item_embeddings[item_id].data = basket_embedding_item[basket_seq_id]
            else:
                    item_embeddings[item_id].data = item_embeddings[item_id]+self.self_attn_item(basket_embedding_item[basket_seq_id])

        #new_price_embeddings
        for price_id, basket_seq_id in price2basket.items():
            if len(basket_seq_id)<2:
                with torch.no_grad():
                    price_embeddings[price_id] = basket_embedding_price[basket_seq_id]
            else:
                with torch.no_grad():
                    price_embeddings[price_id] = price_embeddings[price_id]+self.self_attn_price(basket_embedding_price[basket_seq_id])

        return item_embeddings, price_embeddings

class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, n_category):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category

        self.mat_cp = nn.Parameter(torch.Tensor(self.n_category, 1))
        self.mat_pc = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_cv = nn.Parameter(torch.Tensor(self.n_category, 1))

        self.mat_vp= nn.Parameter(torch.Tensor(self.n_node, 1))
        self.mat_vc= nn.Parameter(torch.Tensor(self.n_node, 1))


        self.a_o_g_i = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gi1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_p = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gp1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_c = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gc1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc2 = nn.Linear(self.emb_size, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)



    def forward(self, adjacency,adjacency_pv,adjacency_vp, adjacency_pc, adjacency_cp, adjacency_cv, adjacency_vc, embedding, pri_emb, cate_emb, single_basket,session_basket):
        for i in range(self.layers):

            item_embeddings = self.inter_gate(self.a_o_g_i, self.b_o_gi1, self.b_o_gi2, embedding, self.get_embedding(adjacency_vp, pri_emb) ,
                 self.get_embedding(adjacency_vc, cate_emb))


            price_embeddings = self.inter_gate(self.a_o_g_p, self.b_o_gp1, self.b_o_gp2, pri_emb,
                                                         self.intra_gate(adjacency_pv, self.mat_pv, embedding),
                                                         self.intra_gate(adjacency_pc, self.mat_pc, cate_emb))

            category_embeddings =  self.inter_gate(self.a_o_g_c, self.b_o_gc1, self.b_o_gc2, cate_emb,
                                                             self.intra_gate(adjacency_cp, self.mat_cp, pri_emb),
                                                             self.intra_gate(adjacency_cv, self.mat_cv, embedding))
            embedding = item_embeddings
            pri_emb = price_embeddings
            cate_emb = category_embeddings

        return embedding , pri_emb,cate_emb



    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        embs = embedding
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings
    def intra_gate(self, adjacency, mat_v, embedding2):
        # attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        matrix = adjacency.to_dense().cuda()
        mat_v = mat_v.expand(mat_v.shape[0], self.emb_size)
        alpha   = torch.mm(mat_v, torch.transpose(embedding2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        alpha = alpha * matrix
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row
        type_embs = torch.mm(alpha, embedding2)
        item_embeddings = type_embs
        return self.dropout70(item_embeddings)
    def inter_gate(self, a_o_g, b_o_g1, b_o_g2, emb_mat1, emb_mat2, emb_mat3):
        all_emb1 = torch.cat([emb_mat1, emb_mat2, emb_mat3], 1)
        gate1 = torch.sigmoid(a_o_g(all_emb1) + b_o_g1(emb_mat2) + b_o_g2(emb_mat3))
        h_embedings = emb_mat1 + gate1 * emb_mat2 + (1 - gate1) * emb_mat3
        return self.dropout50(h_embedings)

class Model(nn.Module):
    def __init__(self, config, numItems,adjacency, adjacency_pv, adjacency_vp,adjacency_pc,adjacency_cp,adjacency_cv,adjacency_vc, n_node, n_price, n_category, lr, layers, l2, beta, dataset,single_basket,session_basket,num_heads=4, emb_size=100, batch_size=100):
        super(Model, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta

        self.single_basket=single_basket
        self.session_basket=session_basket

        self.adjacency = adjacency
        self.adjacency_pv = adjacency_pv
        self.adjacency_vp = adjacency_vp
        self.adjacency_pc = adjacency_pc
        self.adjacency_cp = adjacency_cp
        self.adjacency_cv = adjacency_cv
        self.adjacency_vc = adjacency_vc

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.price_embedding = nn.Embedding(self.n_price, self.emb_size)
        self.category_embedding = nn.Embedding(self.n_category, self.emb_size)

        self.pos_embedding = nn.Embedding(10000, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size, self.n_node, self.n_price, self.n_category)
        self.PowerBasket = PowerBasket(self.emb_size)



        self.w_1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # self_attention
        if emb_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
        # parameters setting
        self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16  the dimension of attention head
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value
        self.query = nn.Linear(self.emb_size, self.emb_size)
        self.key = nn.Linear(self.emb_size, self.emb_size)
        self.value = nn.Linear(self.emb_size, self.emb_size)

        # co-guided networks
        self.w_p_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.u_i_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # gate5 & gate6
        self.w_pi_1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_pi_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_i = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.mlp_m_p_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)
        self.mlp_m_i_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)

        self.mlp_m_p_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.mlp_m_i_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)


        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, price_embedding, session_item, price_seqs, session_len,
                          reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)  # (1，128)
        mask = mask.float().unsqueeze(-1)

        price_embedding = torch.cat([zeros, price_embedding], 0)
        get_pri = lambda i: price_embedding[price_seqs[i]]
        seq_pri = torch.cuda.FloatTensor(mask.shape[0], list(price_seqs.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(price_seqs.shape[0]):
            seq_pri[i] = get_pri(i)  # （100，19，128）

        # self-attention to get price preference
        attention_mask = mask.permute(0, 2, 1).unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(seq_pri)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_pri)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_pri)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)
        sa_result = context_layer.view(*new_context_layer_shape)
        item_pos = torch.tensor(range(1, seq_pri.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(price_seqs)

        item_pos = item_pos * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        price_pre = torch.sum(last_interest, 1)

        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(mask.shape[0], list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)

        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(mask.shape[0], 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        beta = beta * mask
        interest_pre = torch.sum(beta * seq_h, 1)

        p_pre=price_pre
        i_pre=interest_pre



        return i_pre, p_pre


    def transpose_for_scores(self, x, attention_head_size):
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)


    def forward(self,session_item, price_seqs, session_len, mask, reversed_sess_item,samples,sampleLen):

        session_item = trans_to_cuda(torch.Tensor(session_item).long())
        session_len = trans_to_cuda(torch.Tensor(session_len).long())
        price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())


        # Basket-enhanced dynamic hypergraph network
        item_emb_hg=self.embedding.weight
        price_emb_hg=self.price_embedding.weight
        category_emb_hg=self.category_embedding.weight
        sess_emb_hgnn, sess_pri_hgnn = self.generate_sess_emb(item_emb_hg, price_emb_hg, session_item, price_seqs, session_len, reversed_sess_item, mask) # session embeddings in a batch

        # get item-price table return price of items
        v_table = self.adjacency_vp.row##非0元素的值
        temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
        vp_idx = self.adjacency_vp.col[idx]
        item_pri_l = price_emb_hg[vp_idx]

        scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
        scores_price = torch.mm(sess_pri_hgnn, torch.transpose(item_pri_l, 1, 0))
        scores = scores_interest + scores_price
        scores = F.softmax(scores, dim=-1)
        return scores

