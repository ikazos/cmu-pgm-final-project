import numpy as np
import ampligraph
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.distributions.normal import Normal
import sys
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from variationalize import VariationalNormalEmbs, onepass_variational



# GLOVE_PATH = "/usr1/data/sozaki/glove.42B.300d.txt"

print("Is CUDA available?", torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def convert(raw, ent2idx, rel2idx):
    ent_lookup = np.vectorize(lambda ent: ent2idx[ent])
    rel_lookup = np.vectorize(lambda rel: rel2idx[rel])

    sbjs = ent_lookup(raw[:, 0])
    objs = ent_lookup(raw[:, 2])
    rels = rel_lookup(raw[:, 1])

    return (sbjs, objs, rels)



class MyDataset(Dataset):
    def __init__(self, sbjs, objs, rels):
        self.sbjs = torch.tensor(sbjs).to(int)
        self.objs = torch.tensor(objs).to(int)
        self.rels = torch.tensor(rels).to(int)
        assert self.sbjs.shape == self.objs.shape
        assert self.sbjs.shape == self.rels.shape

    def __len__(self):
        return len(self.sbjs)

    def __getitem__(self, k):
        return ( self.sbjs[k], self.objs[k], self.rels[k] )



class NormalEmbs(nn.Module):
    def __init__(self, num_entities, emb_dim, mean, var):
        super(NormalEmbs, self).__init__()

        n = Normal(mean, var)
        weights = n.sample((num_entities, emb_dim))
        self.embs = nn.Embedding.from_pretrained(weights, freeze=False)

    def forward(self, ents):
        return self.embs(ents)



# class GloveEmbs(nn.Module):
#     def __init__(self, weights_path):
#         super(GloveEmbs, self).__init__()

#         with open(weights_path, "rb") as f:
#             weights = torch.tensor(np.load(f)).to(torch.float32)

#         self.embs = nn.Embedding.from_pretrained(weights, freeze=False)

#     def load_word2vec(glove_path):
#         word2vec = dict()
#         with open(glove_path, "r") as f:
#             for line in tqdm(f, total=1917494):
#                 things = line.split()
#                 word = things[0]
#                 vec = np.array(list(map(float, things[1:])))
#                 word2vec[word] = vec

#         return word2vec

#     def precomp(idx2ent, word2vec, weights_path, delims=None):
#         num_ents = len(idx2ent)

#         hits = 0
#         misses = 0

#         weights = np.zeros((num_ents, 300))
#         for idx, ent in enumerate(idx2ent):
#             weight = []

#             if delims == "upper":
#                 spans = []
#                 start = 0
#                 for k, ch in enumerate(ent):
#                     if ch.isupper():
#                         spans.append((start, k))
#                         start = k
#                 spans.append((start, len(ent)))

#                 words = [ ent[start:end] for start, end in spans ]
#             else:
#                 words = [ ent ]
#                 for delim in delims:
#                     words = [ thing.split(delim) for thing in words ]
#                     words = [ item for sublist in words for item in sublist ]

#             for word in words:
#                 word = word.lower()
#                 weight.append(word2vec[word] if word in word2vec else np.zeros((300,)))
#                 hits += word in word2vec
#                 misses += word not in word2vec
#             weights[idx] = np.mean(weight, axis=0)

#         print(f"{hits}/{hits + misses} hits ({float(hits) * 10000 // float(hits + misses) / 100}%)")

#         with open(weights_path, "wb+") as f:
#             np.save(f, weights)

#     def forward(self, ents):
#         return self.embs(ents)



class Bilinear(nn.Module):
    def __init__(self, emb_dim, num_relations):
        super(Bilinear, self).__init__()

        self.bilinear = nn.Bilinear(emb_dim, emb_dim, num_relations)

    def forward(self, sbjs, objs):
        return self.bilinear(sbjs, objs)



# log P (sbj, rel, obj) = log P (rel) + Σ_k log P (sbj_k | rel) + Σ_k log P (obj_k | rel)

class NaiveBayes(nn.Module):
    def __init__(self, emb_dim, num_entities, num_relations, relation_priors):
        super(NaiveBayes, self).__init__()

        self.emb_dim = emb_dim

        # self.mus = torch.randn(num_relations, emb_dim * 2)
        self.register_buffer("mus", torch.randn(num_relations, emb_dim * 2))
        self.mus.requires_grad = True
        #   Use log of sigma to make sure variance is positive
        # self.log_sigmas = torch.randn(num_relations, emb_dim * 2)
        self.register_buffer("log_sigmas", torch.randn(num_relations, emb_dim * 2))
        self.log_sigmas.requires_grad = True

        self.emb_dim = emb_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.relation_priors = relation_priors

        self.sn = Normal(0., 1.)

    def forward(self, sbjs, objs):
        bs = sbjs.shape[0]

        # assert sbjs.shape == (bs, self.emb_dim,)
        # assert objs.shape == (bs, self.emb_dim,)

        sbj_obj = torch.cat((sbjs, objs), dim=-1).unsqueeze(dim=1)
        # assert sbj_obj.shape == (bs, 1, self.emb_dim*2)

        transformed = (sbj_obj - self.mus.unsqueeze(dim=0)) / torch.exp(self.log_sigmas.unsqueeze(dim=0))
        # assert transformed.shape == (bs, self.num_relations, self.emb_dim*2)

        # https://en.wikipedia.org/wiki/Normal_distribution#General_normal_distribution
        # If X ~ N (mu, sigma^2) and Z ~ N (0, 1)
        # Then f_X (x) = 1/sigma f_Z ((x-mu)/sigma)
        # So log f_X (x) = - log sigma + log f_Z ((x-mu)/sigma)

        probs = self.sn.log_prob(transformed) - self.log_sigmas.unsqueeze(dim=0)
        # assert probs.shape == (bs, self.num_relations, self.emb_dim*2)

        probs += self.relation_priors.reshape(1, -1, 1)

        return probs.sum(dim=-1)




# def make_glove_weights_path(dataname):
#     return f"embs/glove-{dataname}-weights.npy"

# def precomp(rawdata, delims, dataname, word2vec):
#     train_raw = rawdata["train"]
#     dev_raw   = rawdata["valid"]
#     test_raw  = rawdata["test"]

#     everything = np.concatenate((train_raw, dev_raw, test_raw))
#     entities = np.unique(np.concatenate([everything[:, 0], everything[:, 2]]))

#     weights_path = make_glove_weights_path(dataname)

#     GloveEmbs.precomp(entities, word2vec, weights_path, delims)



import contextlib

def setup(rawdata, dataname, num_epochs, phases=['pre/retrain','E','EM','pre/retrain'], initial_l2_reg=0.001):
    """ added phase argument"""

    train_raw = rawdata["train"]
    dev_raw   = rawdata["valid"]
    test_raw  = rawdata["test"]

    everything = np.concatenate((train_raw, dev_raw, test_raw))

    entities = np.unique(np.concatenate([everything[:, 0], everything[:, 2]]))
    num_entities = len(entities)

    relations = np.unique(everything[:, 1])
    num_relations = len(relations)

    idx2ent = np.copy(entities)
    ent2idx = { ent: k for k, ent in enumerate(idx2ent) }

    idx2rel = np.copy(relations)
    rel2idx = { rel: k for k, rel in enumerate(idx2rel) }

    train_sbjs, train_objs, train_rels = convert(train_raw, ent2idx, rel2idx)
    dev_sbjs,   dev_objs,   dev_rels   = convert(dev_raw,   ent2idx, rel2idx)
    test_sbjs,  test_objs,  test_rels  = convert(test_raw,  ent2idx, rel2idx)

    """ADDED ENTITY COUNTING FOR REGULARIZATION"""
    e_counts = torch.bincount(torch.Tensor(train_sbjs).int(), minlength=num_entities)
    e_counts += torch.bincount(torch.Tensor(train_objs).int(), minlength=num_entities)
    e_counts = e_counts.float()
    ents_lambdas = initial_l2_reg * e_counts

    train_set = MyDataset(train_sbjs, train_objs, train_rels)
    dev_set   = MyDataset(dev_sbjs,   dev_objs,   dev_rels)
    test_set  = MyDataset(test_sbjs,  test_objs,  test_rels)

    BATCH_SIZE = 64
    # BATCH_SIZE = 512
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_set,   batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


    def onepass(embs, model, optim, is_train, loader):
        if is_train:
            embs.train()
            model.train()
            print("Training...")
        else:
            embs.eval()
            model.eval()
            print("Evaluating...")

        batch_losses = []
        batch_preds = []
        batch_rels = []

        with contextlib.nullcontext() if is_train else torch.no_grad():
            for batch in tqdm(loader):
                sbjs, objs, rels = batch

                sbjs = sbjs.to(device)
                objs = objs.to(device)
                rels = rels.to(device)

                if is_train:
                    optim.zero_grad()

                logits = model(embs(sbjs), embs(objs))
                loss = F.cross_entropy(logits, rels, reduction="none")
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)

                if is_train:
                    loss.sum().backward()

                batch_losses.append(loss.cpu())
                batch_preds.append(preds.cpu())
                batch_rels.append(rels.cpu())

                if is_train:
                    optim.step()

        loss = float(torch.cat(batch_losses).mean())
        preds = torch.cat(batch_preds)
        rels = torch.cat(batch_rels)

        acc = float((preds == rels).to(float).mean())
        f1 = f1_score(rels, preds, average="macro")

        return (acc, f1, loss)

    # Normal embs

    emb_dim = 300
    # normal_embs = NormalEmbs(num_entities, emb_dim, 0., 0.001)
    normal_embs = VariationalNormalEmbs(num_entities, emb_dim, 0., 0.001)
    normal_embs.to(device)

    # normal_bilinear = Bilinear(emb_dim, num_relations)
    # normal_bilinear.to(device)
    priors = torch.zeros(num_relations).to(device)
    normal_nb = NaiveBayes(emb_dim, num_entities, num_relations, priors)
    normal_nb.to(device)

    # normal_optim = optim.Adam(list(normal_embs.parameters()) + list(normal_bilinear.parameters()))
    normal_optim = optim.Adam(list(normal_embs.parameters()) + list(normal_nb.parameters()))

    print("Normal embeddings...")

    for phase in phases:

        training_metrics = []

        for epoch in range(num_epochs):
            # train_acc, train_f1, train_loss = onepass(normal_embs, normal_bilinear, normal_optim, True,  train_loader)
            # dev_acc,   dev_f1,   dev_loss   = onepass(normal_embs, normal_bilinear, None,         False, dev_loader)
            # train_acc, train_f1, train_loss = onepass(normal_embs, normal_nb, normal_optim, True,  train_loader)
            # dev_acc,   dev_f1,   dev_loss   = onepass(normal_embs, normal_nb, None,         False, dev_loader)


            if phase=='EM':
                train_acc, train_f1, train_loss, ents_lambdas = onepass_variational(normal_embs, normal_nb, normal_optim, is_train=True, loader=train_loader,
                                                                        ents_lambdas=ents_lambdas, phase=phase, lambdas_lr=0.8, p=2)
            else:
                train_acc, train_f1, train_loss = onepass_variational(normal_embs, normal_nb, normal_optim, is_train=True, loader=train_loader,
                                                                        ents_lambdas=ents_lambdas, phase=phase, lambdas_lr=0.8, p=2)

            dev_acc,   dev_f1,   dev_loss   = onepass_variational(normal_embs, normal_nb, None, is_train=False, loader=dev_loader,
                                                                    ents_lambdas=ents_lambdas, phase=None, lambdas_lr=0.8, p=2)


            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"\tTraining accuracy: {train_acc:.3f},\tF1: {train_f1:.3f},\tLoss: {float(train_loss):.3f}")
            print(f"\tValidation accuracy: {dev_acc:.3f},\tF1: {dev_f1:.3f},\tLoss: {float(dev_loss):.3f}")

            training_metrics.append(( epoch+1, train_acc, train_f1, train_loss, "train", "normal", dataname ))
            training_metrics.append(( epoch+1, dev_acc, dev_f1, dev_loss, "dev", "normal", dataname ))

        print(ents_lambdas)

        # # Glove embs

        # glove_embs = GloveEmbs(make_glove_weights_path(dataname))
        # glove_embs.to(device)

        # # glove_bilinear = Bilinear(emb_dim, num_relations)
        # # glove_bilinear.to(device)
        # priors = torch.zeros(num_relations).to(device)
        # glove_nb = NaiveBayes(emb_dim, num_entities, num_relations, priors)
        # glove_nb.to(device)

        # # glove_optim = optim.Adam(list(glove_embs.parameters()) + list(glove_bilinear.parameters()))
        # glove_optim = optim.Adam(list(glove_embs.parameters()) + list(glove_nb.parameters()))

        # print("Glove embeddings...")

        # for epoch in range(num_epochs):
        #     # train_acc, train_f1, train_loss = onepass(glove_embs, glove_bilinear, glove_optim, True,  train_loader)
        #     # dev_acc,   dev_f1,   dev_loss   = onepass(glove_embs, glove_bilinear, None,         False, dev_loader)
        #     train_acc, train_f1, train_loss = onepass(glove_embs, glove_nb, glove_optim, True,  train_loader)
        #     dev_acc,   dev_f1,   dev_loss   = onepass(glove_embs, glove_nb, None,         False, dev_loader)

        #     print(f"Epoch {epoch+1}/{num_epochs}")
        #     print(f"\tTraining accuracy: {train_acc:.3f},\tF1: {train_f1:.3f},\tLoss: {float(train_loss):.3f}")
        #     print(f"\tValidation accuracy: {dev_acc:.3f},\tF1: {dev_f1:.3f},\tLoss: {float(dev_loss):.3f}")

        #     training_metrics.append(( epoch+1, train_acc, train_f1, train_loss, "train", "glove", dataname ))
        #     training_metrics.append(( epoch+1, dev_acc, dev_f1, dev_loss, "dev", "glove", dataname ))

    return training_metrics



from ampligraph import datasets

fb15k_237_raw = datasets.load_fb15k_237()
wn18rr_raw    = datasets.load_wn18rr()
yago3_10_raw  = datasets.load_yago3_10()
wn11_raw      = datasets.load_wn11()
fb13_raw      = datasets.load_fb13()



# print("Loading wordvecs...", file=sys.stderr)
# word2vec = GloveEmbs.load_word2vec(GLOVE_PATH)

# print("Precomp fb15k-237...", file=sys.stderr)
# precomp(fb15k_237_raw, "/_",    "fb15k-237", word2vec)

# print("Precomp wn18rr...", file=sys.stderr)
# precomp(wn18rr_raw,    "_",     "wn18rr",    word2vec)

# print("Precomp yago3-10...", file=sys.stderr)
# precomp(yago3_10_raw,  "_",     "yago3-10",  word2vec)

# print("Precomp wn11...", file=sys.stderr)
# precomp(wn11_raw,      "_",     "wn11",      word2vec)

# print("Precomp fb13...", file=sys.stderr)
# precomp(fb13_raw,      "/_",    "fb13",      word2vec)

"""
Precomp fb15k-237...
16354/45182 hits (36.19%)
Precomp wn18rr...
1/40559 hits (0.0%)
Precomp yago3-10...
103055/404510 hits (25.47%)
Precomp wn11...
89641/170537 hits (52.56%)
Precomp fb13...
177684/186649 hits (95.19%)
"""



NUM_EPOCHS = 1

res = [ ( "epoch", "acc", "f1", "loss", "set", "emb", "data" ) ]

print("Training wn18rr...", file=sys.stderr)
res += setup(wn18rr_raw,    "wn18rr",    NUM_EPOCHS)

print("Training on wn11...", file=sys.stderr)
res += setup(wn11_raw,      "wn11",      NUM_EPOCHS)

print("Training on fb15k-237...", file=sys.stderr)
res += setup(fb15k_237_raw, "fb15k-237", NUM_EPOCHS)

print("Training on fb13...", file=sys.stderr)
res += setup(fb13_raw,      "fb13",      NUM_EPOCHS)

print("Training on yago3-10...", file=sys.stderr)
res += setup(yago3_10_raw,  "yago3-10",  NUM_EPOCHS)

import csv
# with open("results-bilinear.csv", "w+") as f:
with open("results-var-nb.csv", "w+") as f:
    writer = csv.writer(f)
    for row in res:
        writer.writerow(row)
