import torch
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm
from collections import Counter

def top_n_elements(a, n):
    counter = Counter(a)
    top_three = counter.most_common(n)
    return [item[0] for item in top_three]
class VTKG(Dataset):
    def __init__(self, data, logger, max_vis_len = -1):
        self.data = data
        self.logger = logger
        self.dir = f"data/{data}/"
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []
        with open(self.dir + "entities.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.ent2id[line.strip()] = idx
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(self.dir + "relations.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.rel2id[line.strip()] = idx
                self.id2rel.append(line.strip())
        self.num_rel = 2 * len(self.rel2id)

        self.train = []
        with open(self.dir + "train.txt") as f:
            for line in f.readlines():
                h,r,t = line.strip().split("\t")
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
                self.train.append((self.ent2id[t], self.rel2id[r] + len(self.rel2id), self.ent2id[h]))

        self.valid = []
        with open(self.dir + "valid.txt") as f:
            for line in f.readlines():
                h,r,t = line.strip().split("\t")
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
                self.valid.append((self.ent2id[t], self.rel2id[r] + len(self.rel2id), self.ent2id[h]))
        self.test = []
        with open(self.dir + "test.txt") as f:
            for line in f.readlines():
                h,r,t = line.strip().split("\t")
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
                self.test.append((self.ent2id[t], self.rel2id[r] + len(self.rel2id), self.ent2id[h]))
        
        self.filter_dict = {}

        for data_split in [self.train, self.valid, self.test]:
            for triplet in data_split:
                h,r,t = triplet
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1,r,t)] = []
                self.filter_dict[(-1,r,t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h,r,-1)] = []
                self.filter_dict[(h,r,-1)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len
        self.gather_vis_feature()
        self.gather_txt_feature()
        self.ent_vis_matrix = None
        self.rel_vis_matrix = None
        self.ent_vis_mask = None
        self.rel_vis_mask = None
        self.ent_txt_matrix = None
        self.rel_txt_matrix = None
        self.vis_feat_size = 0
        self.txt_feat_size = 0
        self.ent2ent, self.leng = self.get_ent2ent()

    def get_ent2ent(self):
        ent_dict_train = {}
        for triplet in self.train:
            h, r, t = triplet
            if h not in ent_dict_train.keys():
                ent_dict_train[h] = []
            if t not in ent_dict_train.keys():
                ent_dict_train[t] = []
            ent_dict_train[h].append(t)
            ent_dict_train[t].append(h)
        ent_dict_train = {k: list(set(v)) for k, v in ent_dict_train.items()}
        leng = {}
        for key in ent_dict_train:
            value = ent_dict_train[key]
            leng[key] = len(value) + 1
            if len(value) < 10:
                ent_dict_train[key] = value + [self.num_ent] * (10 - len(value))
            else:
                ent_dict_train[key] = value[:10]
                leng[key] = 11
        for i in range(self.num_ent):
            if i not in ent_dict_train.keys():
                ent_dict_train[i] = [self.num_ent] * 10
            if i not in leng.keys():
                leng[i] = 1
        return ent_dict_train, leng
    def get_neigh_rel(self):
        rel_dict_train = {}
        for triplet in self.train:
            h, r, t = triplet
            if r not in rel_dict_train.keys():
                rel_dict_train[r] = []
            rel_dict_train[r].append(h)
            rel_dict_train[r].append(t)
        # 统计a的每个key中出现次数最多的三个值
        rel_dict_three = torch.zeros([self.num_rel, 3])
        for j in rel_dict_train:
            rel_dict_three[j] = torch.IntTensor(top_n_elements(rel_dict_train[j], 3))
        return rel_dict_three
    def get_neigh_ent(self):
        ent_dict_train = {}
        for triplet in self.train:
            h, r, t = triplet
            if h not in ent_dict_train.keys():
                ent_dict_train[h] = []
            if t not in ent_dict_train.keys():
                ent_dict_train[t] = []
            ent_dict_train[h].append(r)
            ent_dict_train[t].append(r)
        # 统计a的每个key中出现次数最多的三个值
        ent_dict_three = torch.ones([self.num_ent, 3]) * 100000
        for j in ent_dict_train:
            tmp = list(set(ent_dict_train[j]))
            n = len(tmp)
            if n > 3:
                ent_dict_three[j] = torch.IntTensor(top_n_elements(ent_dict_train[j], 3))
            elif n==3:
                ent_dict_three[j] = torch.IntTensor(tmp)
            elif n==2:
                ent_dict_three[j][0] = int(tmp[0])
                ent_dict_three[j][1] = int(tmp[1])
            else:
                ent_dict_three[j][0] = int(tmp[0])
        return ent_dict_three
    def sort_vis_features(self, item = 'entity'):
        if item == 'entity':
            vis_feats = torch.load(self.dir + 'visual_features_ent.pt')
        elif item == 'relation':
            vis_feats = torch.load(self.dir + 'visual_features_rel.pt')
        else:
            raise NotImplementedError
        
        sorted_vis_feats = {}
        for obj in tqdm(vis_feats):
            if item == 'entity' and obj not in self.ent2id:
                continue
            if item == 'relation' and obj not in self.rel2id:
                continue
            num_feats = len(vis_feats[obj])
            sim_val = torch.zeros(num_feats).cuda()
            iterate = tqdm(range(num_feats)) if num_feats > 1000 else range(num_feats)
            cudaed_feats = vis_feats[obj].cuda()
            for i in iterate:
                sims = torch.inner(cudaed_feats[i], cudaed_feats[i:])
                sim_val[i:] += sims
                sim_val[i] += sims.sum()-torch.inner(cudaed_feats[i], cudaed_feats[i])
            sorted_vis_feats[obj] = vis_feats[obj][torch.argsort(sim_val, descending = True)]

        if item == 'entity':
            torch.save(sorted_vis_feats, self.dir+ "visual_features_ent_sorted.pt")
        else:
            torch.save(sorted_vis_feats, self.dir+ "visual_features_rel_sorted.pt")
        
        return sorted_vis_feats

    def gather_vis_feature(self):
        if os.path.isfile(self.dir + 'visual_features_ent_sorted.pt'):
            self.logger.info("Found sorted entity visual features!")
            self.ent2vis = torch.load(self.dir + 'visual_features_ent_sorted.pt')
        elif os.path.isfile(self.dir + 'visual_features_ent.pt'):
            self.logger.info("Entity visual features are not sorted! sorting...")
            self.ent2vis = self.sort_vis_features(item = 'entity')
        else:
            self.logger.info("Entity visual features are not found!")
            self.ent2vis = {}
        
        if os.path.isfile(self.dir + 'visual_features_rel_sorted.pt'):
            self.logger.info("Found sorted relation visual features!")
            self.rel2vis = torch.load(self.dir + 'visual_features_rel_sorted.pt')
        elif os.path.isfile(self.dir + 'visual_features_rel.pt'):
            self.logger.info("Relation visual feature are not sorted! sorting...")
            self.rel2vis = self.sort_vis_features(item = 'relation')
        else:
            self.logger.info("Relation visual features are not found!")
            self.rel2vis = {}
        
        self.vis_feat_size = len(self.ent2vis[list(self.ent2vis.keys())[0]][0])

        total_num = 0
        if self.max_vis_len_ent != -1:
            for ent_name in self.ent2vis:
                num_feats = len(self.ent2vis[ent_name])
                total_num += num_feats
                self.ent2vis[ent_name] = self.ent2vis[ent_name][:self.max_vis_len_ent]
            for rel_name in self.rel2vis:
                self.rel2vis[rel_name] = self.rel2vis[rel_name][:self.max_vis_len_rel]
        else:
            for ent_name in self.ent2vis:
                num_feats = len(self.ent2vis[ent_name])
                total_num += num_feats
                if self.max_vis_len_ent < len(self.ent2vis[ent_name]):
                    self.max_vis_len_ent = len(self.ent2vis[ent_name])
            self.max_vis_len_ent = max(self.max_vis_len_ent, 0)
            for rel_name in self.rel2vis:
                if self.max_vis_len_rel < len(self.rel2vis[rel_name]):
                    self.max_vis_len_rel = len(self.rel2vis[rel_name])
            self.max_vis_len_rel = max(self.max_vis_len_rel, 0)
        self.ent_vis_mask = torch.full((self.num_ent, self.max_vis_len_ent), True).cuda()
        self.ent_vis_matrix = torch.zeros((self.num_ent, self.max_vis_len_ent, self.vis_feat_size)).cuda()
        self.rel_vis_mask = torch.full((self.num_rel, self.max_vis_len_rel), True).cuda()
        self.rel_vis_matrix = torch.zeros((self.num_rel, self.max_vis_len_rel, 3*self.vis_feat_size)).cuda()

        
        for ent_name in self.ent2vis:
            ent_id = self.ent2id[ent_name]
            num_feats = len(self.ent2vis[ent_name])
            self.ent_vis_mask[ent_id, :num_feats] = False
            self.ent_vis_matrix[ent_id, :num_feats] = self.ent2vis[ent_name]

        for rel_name in self.rel2vis:
            rel_id = self.rel2id[rel_name]
            num_feats = len(self.rel2vis[rel_name])
            self.rel_vis_mask[rel_id, :num_feats] = False
            self.rel_vis_matrix[rel_id, :num_feats] = self.rel2vis[rel_name]#rel有东西的时候可能需要改

    def gather_txt_feature(self):

        self.ent2txt = torch.load(self.dir + 'textual_features_ent.pt')
        self.rel2txt = torch.load(self.dir + 'textual_features_rel.pt')
        self.txt_feat_size = len(self.ent2txt[self.id2ent[0]])

        self.ent_txt_matrix = torch.zeros((self.num_ent, self.txt_feat_size)).cuda()
        self.rel_txt_matrix = torch.zeros((self.num_rel, self.txt_feat_size)).cuda()

        for ent_name in self.ent2id:
            self.ent_txt_matrix[self.ent2id[ent_name]] = self.ent2txt[ent_name]

        for rel_name in self.rel2id:
            self.rel_txt_matrix[self.rel2id[rel_name]] = self.rel2txt[rel_name]
            self.rel_txt_matrix[self.rel2id[rel_name] + len(self.rel2id)] = self.rel2txt[rel_name]

    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        h,r,t = self.train[idx]
        # if random.random() < 0.5:
        #     masked_triplet = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
        #     label = h
        # else:
        #     masked_triplet = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
        #     label = t
        masked_triplet = [h, r, t]
        ent2ent = torch.tensor(self.ent2ent[h])

        leng = self.leng[h]
        if leng <=2:
            ent2ent[0] = self.num_ent
            leng = 1
        else:
            index = leng - 2
            tmp = ent2ent[index].clone()
            # ent2ent[index] = self.num_ent
            indices = (ent2ent == t).nonzero(as_tuple=True)[0]
            if indices.shape[0]!= 0:
                ent2ent[indices[0]] = tmp
                ent2ent[index] = self.num_ent
                leng = leng - 1
        return torch.tensor(masked_triplet), ent2ent, leng


class VTKGTime(Dataset):
    def __init__(self, data, logger, temporal, max_vis_len=-1):
        self.data = data
        self.logger = logger
        self.dir = f"data/{data}/"
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []
        self.temporal = temporal
        with open(self.dir + "entities.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.ent2id[line.strip()] = idx
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(self.dir + "relations.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.rel2id[line.strip()] = idx
                self.id2rel.append(line.strip())
        self.num_rel = 2 * len(self.rel2id)

        with open(self.dir + "times.txt") as f:
            self.time = []
            for line in f.readlines():
                self.time.append(line.strip())
        self.time2id = {time: idx for idx, time in enumerate(self.time)}
        self.id2time = {idx: time for idx, time in enumerate(self.time)}
        self.num_time = len(self.time)
        self.train = []
        with open(self.dir + "train.txt") as f:
            for line in f.readlines():
                h, r, t, time = line.strip().split("\t")
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t], self.time2id[time]))
                self.train.append((self.ent2id[t], self.rel2id[r] + len(self.rel2id), self.ent2id[h], self.time2id[time]))

        self.valid = []
        with open(self.dir + "valid.txt") as f:
            for line in f.readlines():
                h, r, t, time = line.strip().split("\t")
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t], self.time2id[time]))
                self.valid.append((self.ent2id[t], self.rel2id[r] + len(self.rel2id), self.ent2id[h], self.time2id[time]))
        self.test = []
        with open(self.dir + "test.txt") as f:
            for line in f.readlines():
                h, r, t, time = line.strip().split("\t")
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t], self.time2id[time]))
                self.test.append((self.ent2id[t], self.rel2id[r] + len(self.rel2id), self.ent2id[h], self.time2id[time]))

        self.filter_dict = {}

        for data_split in [self.train, self.valid, self.test]:
            for triplet in data_split:
                h, r, t, time = triplet
                if (-1, r, t, time) not in self.filter_dict:
                    self.filter_dict[(-1, r, t, time)] = []
                self.filter_dict[(-1, r, t, time)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1, time)] = []
                self.filter_dict[(h, r, -1, time)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len
        # self.gather_vis_feature()
        # self.gather_txt_feature()
        self.ent_vis_matrix = None
        self.rel_vis_matrix = None
        self.ent_vis_mask = None
        self.rel_vis_mask = None
        self.ent_txt_matrix = None
        self.rel_txt_matrix = None
        self.vis_feat_size = 0
        self.txt_feat_size = 0
        self.ent2ent, self.leng = self.get_ent2ent()

    def get_ent2ent(self):
        ent_dict_train = {}
        for triplet in self.train:
            h, r, t, time = triplet
            if h not in ent_dict_train.keys():
                ent_dict_train[h] = []
            if t not in ent_dict_train.keys():
                ent_dict_train[t] = []
            ent_dict_train[h].append(t)
            ent_dict_train[t].append(h)
        ent_dict_train = {k: list(set(v)) for k, v in ent_dict_train.items()}
        leng = {}
        for key in ent_dict_train:
            value = ent_dict_train[key]
            leng[key] = len(value) + 1
            if len(value) < 10:
                ent_dict_train[key] = value + [self.num_ent] * (10 - len(value))
            else:
                ent_dict_train[key] = value[:10]
                leng[key] = 11
        for i in range(self.num_ent):
            if i not in ent_dict_train.keys():
                ent_dict_train[i] = [self.num_ent] * 10
            if i not in leng.keys():
                leng[i] = 1
        return ent_dict_train, leng

    def get_neigh_rel(self):
        rel_dict_train = {}
        for triplet in self.train:
            h, r, t, time = triplet
            if r not in rel_dict_train.keys():
                rel_dict_train[r] = []
            rel_dict_train[r].append(h)
            rel_dict_train[r].append(t)
        # 统计a的每个key中出现次数最多的三个值
        rel_dict_three = torch.zeros([self.num_rel, 2])
        for j in rel_dict_train:
            rel_dict_three[j] = torch.IntTensor(top_n_elements(rel_dict_train[j], 2))
        return rel_dict_three

    def get_neigh_ent(self):
        ent_dict_train = {}
        for triplet in self.train:
            h, r, t, time = triplet
            if h not in ent_dict_train.keys():
                ent_dict_train[h] = []
            if t not in ent_dict_train.keys():
                ent_dict_train[t] = []
            ent_dict_train[h].append(r)
            ent_dict_train[t].append(r)
        # 统计a的每个key中出现次数最多的三个值
        # ent_dict_three = torch.ones([self.num_ent, 3]) * 100000
        # for j in ent_dict_train:
        #     tmp = list(set(ent_dict_train[j]))
        #     n = len(tmp)
        #     if n > 3:
        #         ent_dict_three[j] = torch.IntTensor(top_n_elements(ent_dict_train[j], 3))
        #     elif n == 3:
        #         ent_dict_three[j] = torch.IntTensor(tmp)
        #     elif n == 2:
        #         ent_dict_three[j][0] = int(tmp[0])
        #         ent_dict_three[j][1] = int(tmp[1])
        #     else:
        #         ent_dict_three[j][0] = int(tmp[0])
        ent_dict_three = torch.ones([self.num_ent, 2]) * 0
        for j in ent_dict_train:
            tmp = list(set(ent_dict_train[j]))
            n = len(tmp)
            if n > 2:
                ent_dict_three[j] = torch.IntTensor(top_n_elements(ent_dict_train[j], 2))
            elif n == 2:
                ent_dict_three[j][0] = int(tmp[0])
                ent_dict_three[j][1] = int(tmp[1])
            else:
                ent_dict_three[j][0] = int(tmp[0])
        return ent_dict_three

    def sort_vis_features(self, item='entity'):
        if item == 'entity':
            vis_feats = torch.load(self.dir + 'visual_features_ent.pt')
        elif item == 'relation':
            vis_feats = torch.load(self.dir + 'visual_features_rel.pt')
        else:
            raise NotImplementedError

        sorted_vis_feats = {}
        for obj in tqdm(vis_feats):
            if item == 'entity' and obj not in self.ent2id:
                continue
            if item == 'relation' and obj not in self.rel2id:
                continue
            num_feats = len(vis_feats[obj])
            sim_val = torch.zeros(num_feats).cuda()
            iterate = tqdm(range(num_feats)) if num_feats > 1000 else range(num_feats)
            cudaed_feats = vis_feats[obj].cuda()
            for i in iterate:
                sims = torch.inner(cudaed_feats[i], cudaed_feats[i:])
                sim_val[i:] += sims
                sim_val[i] += sims.sum() - torch.inner(cudaed_feats[i], cudaed_feats[i])
            sorted_vis_feats[obj] = vis_feats[obj][torch.argsort(sim_val, descending=True)]

        if item == 'entity':
            torch.save(sorted_vis_feats, self.dir + "visual_features_ent_sorted.pt")
        else:
            torch.save(sorted_vis_feats, self.dir + "visual_features_rel_sorted.pt")

        return sorted_vis_feats

    def gather_vis_feature(self):
        if os.path.isfile(self.dir + 'visual_features_ent_sorted.pt'):
            self.logger.info("Found sorted entity visual features!")
            self.ent2vis = torch.load(self.dir + 'visual_features_ent_sorted.pt')
        elif os.path.isfile(self.dir + 'visual_features_ent.pt'):
            self.logger.info("Entity visual features are not sorted! sorting...")
            self.ent2vis = self.sort_vis_features(item='entity')
        else:
            self.logger.info("Entity visual features are not found!")
            self.ent2vis = {}

        if os.path.isfile(self.dir + 'visual_features_rel_sorted.pt'):
            self.logger.info("Found sorted relation visual features!")
            self.rel2vis = torch.load(self.dir + 'visual_features_rel_sorted.pt')
        elif os.path.isfile(self.dir + 'visual_features_rel.pt'):
            self.logger.info("Relation visual feature are not sorted! sorting...")
            self.rel2vis = self.sort_vis_features(item='relation')
        else:
            self.logger.info("Relation visual features are not found!")
            self.rel2vis = {}

        self.vis_feat_size = len(self.ent2vis[list(self.ent2vis.keys())[0]][0])

        total_num = 0
        if self.max_vis_len_ent != -1:
            for ent_name in self.ent2vis:
                num_feats = len(self.ent2vis[ent_name])
                total_num += num_feats
                self.ent2vis[ent_name] = self.ent2vis[ent_name][:self.max_vis_len_ent]
            for rel_name in self.rel2vis:
                self.rel2vis[rel_name] = self.rel2vis[rel_name][:self.max_vis_len_rel]
        else:
            for ent_name in self.ent2vis:
                num_feats = len(self.ent2vis[ent_name])
                total_num += num_feats
                if self.max_vis_len_ent < len(self.ent2vis[ent_name]):
                    self.max_vis_len_ent = len(self.ent2vis[ent_name])
            self.max_vis_len_ent = max(self.max_vis_len_ent, 0)
            for rel_name in self.rel2vis:
                if self.max_vis_len_rel < len(self.rel2vis[rel_name]):
                    self.max_vis_len_rel = len(self.rel2vis[rel_name])
            self.max_vis_len_rel = max(self.max_vis_len_rel, 0)
        self.ent_vis_mask = torch.full((self.num_ent, self.max_vis_len_ent), True).cuda()
        self.ent_vis_matrix = torch.zeros((self.num_ent, self.max_vis_len_ent, self.vis_feat_size)).cuda()
        self.rel_vis_mask = torch.full((self.num_rel, self.max_vis_len_rel), True).cuda()
        self.rel_vis_matrix = torch.zeros((self.num_rel, self.max_vis_len_rel, 3 * self.vis_feat_size)).cuda()

        for ent_name in self.ent2vis:
            ent_id = self.ent2id[ent_name]
            num_feats = len(self.ent2vis[ent_name])
            self.ent_vis_mask[ent_id, :num_feats] = False
            self.ent_vis_matrix[ent_id, :num_feats] = self.ent2vis[ent_name]

        for rel_name in self.rel2vis:
            rel_id = self.rel2id[rel_name]
            num_feats = len(self.rel2vis[rel_name])
            self.rel_vis_mask[rel_id, :num_feats] = False
            self.rel_vis_matrix[rel_id, :num_feats] = self.rel2vis[rel_name]  # rel有东西的时候可能需要改

    def gather_txt_feature(self):

        self.ent2txt = torch.load(self.dir + 'textual_features_ent.pt')
        self.rel2txt = torch.load(self.dir + 'textual_features_rel.pt')
        self.txt_feat_size = len(self.ent2txt[self.id2ent[0]])

        self.ent_txt_matrix = torch.zeros((self.num_ent, self.txt_feat_size)).cuda()
        self.rel_txt_matrix = torch.zeros((self.num_rel, self.txt_feat_size)).cuda()

        for ent_name in self.ent2id:
            self.ent_txt_matrix[self.ent2id[ent_name]] = self.ent2txt[ent_name]

        for rel_name in self.rel2id:
            self.rel_txt_matrix[self.rel2id[rel_name]] = self.rel2txt[rel_name]
            self.rel_txt_matrix[self.rel2id[rel_name] + len(self.rel2id)] = self.rel2txt[rel_name]

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        h, r, t, time = self.train[idx]
        # if random.random() < 0.5:
        #     masked_triplet = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
        #     label = h
        # else:
        #     masked_triplet = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
        #     label = t
        masked_triplet = [h, r, t, time]
        ent2ent = torch.tensor(self.ent2ent[h])

        leng = self.leng[h]
        if leng <= 2:
            ent2ent[0] = self.num_ent
            leng = 1
        else:
            index = leng - 2
            tmp = ent2ent[index].clone()
            # ent2ent[index] = self.num_ent
            indices = (ent2ent == t).nonzero(as_tuple=True)[0]
            if indices.shape[0] != 0:
                ent2ent[indices[0]] = tmp
                ent2ent[index] = self.num_ent
                leng = leng - 1
        return torch.tensor(masked_triplet), ent2ent, leng
