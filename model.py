import torch
import torch.nn as nn
import json
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import LSTM


class MIN(nn.Module):
    def __init__(self, dim_str, tensor_n):
        super(MIN, self).__init__()
        self.drop = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(dim_str, 2048)
        self.linear2 = nn.Linear(2048, dim_str)
        self.norm1 = nn.LayerNorm(dim_str)
        self.norm2 = nn.LayerNorm(dim_str)
        self.dropout = nn.Dropout(0.1)
        self.w_in = nn.Parameter(torch.rand(dim_str, dim_str, tensor_n))
        nn.init.xavier_uniform_(self.w_in.data)
        self.w_out = nn.Parameter(torch.rand(dim_str, dim_str, tensor_n))
        nn.init.xavier_uniform_(self.w_out.data)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()
    def _sa_block(self, x):
        x = x.permute(0, 2, 1)
        e_num = x.shape[0]
        x = torch.cat([self.tt_product_nd(x[:e_num // 2, :], self.w_in),
                           self.tt_product_nd(x[e_num // 2:, :], self.w_out)])
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        return x

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)
    def tt_product_nd(self, x_new, w_new):
        """
        n分量张量积通用计算
        输入形状:
        - x_new: [batch_size, in_dim, n]
        - w_new: [in_dim, out_dim, n]
        输出形状:
        - h: [batch_size, out_dim, n]
        """
        batch_size, in_dim, n = x_new.shape
        out_dim = w_new.shape[1]
        h = torch.zeros([batch_size, out_dim, n], device=x_new.device)

        for k in range(n):
            for i in range(n):
                # 计算w分量索引: (k - i) mod n
                j = (k - i) % n
                h[:, :, k] += torch.matmul(x_new[:, :, i], w_new[:, :, j])

        return h
    def forward(self, ent_seq):
        ent_seq = ent_seq + self._sa_block(self.norm1(ent_seq))
        ent_embs = ent_seq + self._ff_block(self.norm2(ent_seq))
        return ent_embs

class DTME(nn.Module):
    def __init__(self, num_ent, num_rel, num_time, ent_vis, rel_vis, dim_vis, ent_txt, rel_txt, dim_txt, ent_vis_mask,
                 rel_vis_mask, \
                 dim_str, num_head, dim_hid, num_layer_enc_ent, num_layer_enc_rel, num_layer_dec, rel_neigh, ent_neigh,
                 phm, dropout=0.1, \
                 emb_dropout=0.6, img_dropout=0.1, txt_dropout=0.1, vid_dropout=0.1, aud_dropout=0.1, time_dropout=0.1,):
        super(DTME, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.num_time = num_time
        self.phm = phm
        self.rank = dim_str // 2
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1, dim_str))
        self.time_embeddings = nn.Embedding(num_time, dim_str)
        self.device = torch.device('cuda')
        self.dtype = torch.float32
        self.rel_neigh = rel_neigh.long()
        self.ent_neigh = ent_neigh.long()
        self.pi = 3.14159265358979323846

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))

        with open('data/focus/processed_entities.json', 'r', encoding='utf-8') as f:
            self.entity_multimodal = json.load(f)
        self.entity_multimodal = self.dict_to_matrices(self.entity_multimodal)
        with open('data/focus/processed_relations.json', 'r', encoding='utf-8') as f:
            self.relation_multimodal = json.load(f)


        self.relation_multimodal = self.dict_to_matrices(self.relation_multimodal)
        self.relation_multimodal['text'] = self.relation_multimodal['text'].repeat(2, 1)
        self.relation_multimodal['video'] = self.relation_multimodal['video'].repeat(2, 1)
        self.relation_multimodal['image'] = self.relation_multimodal['image'].repeat(2, 1)
        self.relation_multimodal['audio'] = self.relation_multimodal['audio'].repeat(2, 1)

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.img_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)
        self.vid_ln = nn.LayerNorm(dim_str)
        self.aud_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p=emb_dropout)
        self.imgdr = nn.Dropout(p=img_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)
        self.viddr = nn.Dropout(p=vid_dropout)
        self.audr = nn.Dropout(p=aud_dropout)
        self.timedr = nn.Dropout(p=time_dropout)
        self.proj_img = nn.Linear(768, dim_str)
        self.proj_txt = nn.Linear(768, dim_str)
        self.proj_aud = nn.Linear(128, dim_str)
        self.proj_vid = nn.Linear(768, dim_str)

        self.moci_e = MIN(dim_str, 41)
        self.moci_r = MIN(dim_str, 41)


        alpha=0.1#select the parameter
        gamma=0.1#select the parameter
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_img_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vid_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_aud_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_img_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vid_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_aud_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))

        nn.init.xavier_uniform_(self.proj_img.weight)
        nn.init.xavier_uniform_(self.proj_txt.weight)
        nn.init.xavier_uniform_(self.proj_aud.weight)
        nn.init.xavier_uniform_(self.proj_vid.weight)

        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_img_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_vid_ent)
        nn.init.xavier_uniform_(self.pos_aud_ent)

        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_img_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_vid_rel)
        nn.init.xavier_uniform_(self.pos_aud_rel)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        self.proj_img.bias.data.zero_()
        self.proj_txt.bias.data.zero_()
        self.proj_aud.bias.data.zero_()
        self.proj_vid.bias.data.zero_()
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)


    def dict_to_matrices(self, processed_data):
        """
        将处理后的多模态字典转换为各模态的矩阵
        :param processed_data: 处理后的多模态字典 {key: {'text':[], 'image':[], ...}}
        :return: 各模态的矩阵字典 {'text': matrix, 'image': matrix, ...}
        """
        # 保存原始键顺序
        self.keys = list(processed_data.keys())
        num_items = len(self.keys)
        modal_matrices = {
            'text': None,
            'image': None,
            'video': None,
            'audio': None
        }
        # 初始化各模态矩阵
        sample = processed_data[self.keys[0]]
        dims = {modal: len(sample[modal]) for modal in modal_matrices}
        for modal in modal_matrices.keys():
            # 获取向量维度（假设所有向量维度相同）
            modal_matrices[modal] = torch.empty(
                (num_items, dims[modal]),
                dtype=self.dtype,
                # device=self.device
            ).cuda()

        # 填充矩阵（保持原始顺序）
        with tqdm(total=num_items, desc=f"Converting to Tensors ({self.device})") as pbar:
            for idx, (_, features) in enumerate(processed_data.items()):
                for modal, tensor in modal_matrices.items():
                    # 直接从Python列表创建Tensor并移到目标设备
                    tensor[idx] = torch.tensor(
                        features[modal],
                        dtype=self.dtype,
                        # device=self.device
                    ).cuda()
                pbar.update(1)

        return modal_matrices
    def forward(self):
        text_entity_embed = self.entity_multimodal['text']
        image_entity_embed = self.entity_multimodal['image']
        video_entity_embed = self.entity_multimodal['video']
        audio_entity_embed = self.entity_multimodal['audio']

        text_relation_embed = self.relation_multimodal['text']
        image_relation_embed = self.relation_multimodal['image']
        video_relation_embed = self.relation_multimodal['video']
        audio_relation_embed = self.relation_multimodal['audio']

        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rel_tkn = self.rel_token.tile(self.num_rel, 1, 1)

        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        rep_ent_img = self.imgdr(
            self.img_ln(self.proj_img(image_entity_embed))).unsqueeze(
            1) + self.pos_img_ent  # 41105,1,768 -> 41105,1,256
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_txt(text_entity_embed))).unsqueeze(1) + self.pos_txt_ent
        rep_ent_vid = self.viddr(self.vid_ln(self.proj_vid(video_entity_embed))).unsqueeze(1) + self.pos_vid_ent
        rep_ent_aud = self.audr(self.aud_ln(self.proj_aud(audio_entity_embed))).unsqueeze(1) + self.pos_aud_ent
        neigh_ent_indices = self.ent_neigh  # shape: [n_ent, 2]

        neigh_ent1_seq = ent_tkn[neigh_ent_indices[:, 0]]

        neigh_ent1_modalities = neigh_ent1_seq[:, 1:]  # [n_ent, 5, dim]

        neigh_ents_combined = neigh_ent1_modalities
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_img, rep_ent_txt, rep_ent_vid, rep_ent_aud, neigh_ents_combined], dim=1)  # 41105,4,256

        ent_embs = self.moci_e(ent_seq)

        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings)) + self.pos_str_rel
        rep_rel_img = self.imgdr(self.img_ln(self.proj_img(image_relation_embed))).unsqueeze(1) + self.pos_img_rel
        rep_rel_txt = self.txtdr(self.txt_ln(self.proj_txt(text_relation_embed))).unsqueeze(1) + self.pos_txt_rel
        rep_rel_vid = self.viddr(self.vid_ln(self.proj_vid(video_relation_embed))).unsqueeze(1) + self.pos_vid_rel
        rep_rel_aud = self.audr(self.aud_ln(self.proj_aud(audio_relation_embed))).unsqueeze(1) + self.pos_aud_rel

        neigh_ent_indices = self.rel_neigh  # shape: [n_rel, 2]

        neigh_ent1_seq = ent_seq[neigh_ent_indices[:, 0]]
        neigh_ent2_seq = ent_seq[neigh_ent_indices[:, 1]]
        neigh_ent1_modalities = neigh_ent1_seq[:, 1:]  # [n_rel, 5, dim]
        neigh_ent2_modalities = neigh_ent2_seq[:, 1:]  # [n_rel, 5, dim]

        neigh_ents_combined = torch.cat([neigh_ent1_modalities, neigh_ent2_modalities], dim=1)

        rel_seq = torch.cat([rel_tkn, rep_rel_str, rep_rel_img, rep_rel_txt, rep_rel_vid, rep_rel_aud, neigh_ents_combined], dim=1)
        rel_embs = self.moci_r(rel_seq)
        return ent_embs.permute(1,0,2)[:6,:,:], rel_embs.permute(1,0,2)[:6,:,:]


    def scoring(self, h, t, r):
        return torch.sum(h * t * r, 1, False)

    def TADistMult(self,lhs, rel, rhs, time, ent_embs, rel_embs):
        pos_h_e = lhs
        pos_rseq_e = self.get_rseq(rel, time)

        pos = pos_h_e * pos_rseq_e @ ent_embs.transpose(0, 1)
        return pos

    def get_rseq(self, r, tem):
        bs = tem.shape[0]  # batch size
        r_e = r.view(bs, 1, -1)
        token_e = tem.view(bs, 1, -1)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0][:, -1, :]
        rseq_e = hidden_tem

        return rseq_e
    def Tcomplex(self,lhs, rel, rhs, time, ent_embs, rel_embs):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)@ ent_embs.t()
    def TcompoundE(self,lhs, rel, rhs, time, ent_embs, rel_embs):

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        right = ent_embs
        right = right[:, :self.rank], right[:, self.rank:]
        rt = (rel[0] + time[0]) * time[1], rel[1]
        return ((lhs[0] + rt[1]) * rt[0] ) @ right[0].t()


    def score(self, emb_ent, emb_rel, triplets):
        lhs_mm = emb_ent[0][triplets[:, 0]]
        rhs_mm = emb_ent[0][triplets[:, 2]]
        lhs_str = emb_ent[1][triplets[:, 0]]
        rhs_str = emb_ent[1][triplets[:, 2]]
        lhs_img = emb_ent[2][triplets[:, 0]]
        rhs_img = emb_ent[2][triplets[:, 2]]
        lhs_txt = emb_ent[3][triplets[:, 0]]
        rhs_txt = emb_ent[3][triplets[:, 2]]
        lhs_vid = emb_ent[4][triplets[:, 0]]
        rhs_vid = emb_ent[4][triplets[:, 2]]
        lhs_aud = emb_ent[5][triplets[:, 0]]
        rhs_aud = emb_ent[5][triplets[:, 2]]

        rel_mm = emb_rel[0][triplets[:, 1]]
        rel_str = emb_rel[1][triplets[:, 1]]
        rel_img = emb_rel[2][triplets[:, 1]]
        rel_txt = emb_rel[3][triplets[:, 1]]
        rel_vid = emb_rel[4][triplets[:, 1]]
        rel_aud = emb_rel[5][triplets[:, 1]]
        time = self.time_embeddings(triplets[:, 3])
        time = self.timedr(time)

        scores_mm = self.Tcomplex(lhs_mm, rel_mm, rhs_mm, time, emb_ent[0], emb_rel)
        scores_str = self.Tcomplex(lhs_str, rel_str, rhs_str, time, emb_ent[1], emb_rel)
        scores_img = self.Tcomplex(lhs_img, rel_img, rhs_img, time, emb_ent[2], emb_rel)
        scores_txt = self.Tcomplex(lhs_txt, rel_txt, rhs_txt, time, emb_ent[3], emb_rel)
        scores_vid = self.Tcomplex(lhs_vid, rel_vid, rhs_vid, time, emb_ent[4], emb_rel)
        scores_aud = self.Tcomplex(lhs_aud, rel_aud, rhs_aud, time, emb_ent[5], emb_rel)
        return scores_mm, scores_str, scores_img, scores_txt, scores_vid, scores_aud