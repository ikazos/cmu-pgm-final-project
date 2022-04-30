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
from torch.nn import Parameter


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

class VariationalEntityEmbs(nn.Module):
    def __init__(self, num_entities, emb_dim, mean, var):
        super(VariationalEntityEmbs, self).__init__()

        n = torch.distributions.Normal(mean, var)
        weights_means  = n.sample((num_entities, emb_dim))
        weights_logstd = n.sample((num_entities, emb_dim))

        self.embs_means = nn.Embedding.from_pretrained(weights_means, freeze=False)
        self.embs_logstd = nn.Embedding.from_pretrained(weights_logstd, freeze=False)
        self.N = torch.distributions.Normal(0, 1)
        self.p = 2
        self.emb_dim = emb_dim
        self.num_entities = num_entities

    def forward(self, ents, is_train=True, only_means=True):

        ents_means = self.embs_means(ents)

        if is_train and only_means:
            return ents_means, torch.norm(ents_means,p=self.p, dim=(1,))
        elif not is_train:
            return ents_means
        else:
            ents_logstd = self.embs_logstd(ents)
            # entropic_loss = torch.sum(ents_logstd) #return sum of log standard deviations; this is the entropy term of a gaussian
            entropic_loss = torch.sum(self.embs_logstd.weight) #add loss for ALL logstddevs instead of those in the batch
            ents_logstd = torch.exp(ents_logstd)

            ents_noise = self.N.sample(ents_logstd.shape).to(ents.device) #not actually noise, reparametrization trick
            ents = ents_means + ents_logstd * ents_noise
            reg_loss_ents = torch.norm(ents,p=self.p, dim=(1,)) #need to return norm of embeddings with noise to compute L2 loss

            return ents, reg_loss_ents, entropic_loss

class VariationalBilinear(nn.Module):
    def __init__(self, num_relations, emb_dim, mean, var):
        super(VariationalBilinear, self).__init__()

        self.bilinear_means = nn.init.normal_(Parameter(torch.empty((num_relations, emb_dim, emb_dim))), mean, var**0.5)
        self.bilinear_logstd =  nn.init.normal_(Parameter(torch.empty((num_relations, emb_dim, emb_dim))), mean, var**0.5)

        self.N = torch.distributions.Normal(0, 1)
        self.p = 2
        self.emb_dim = emb_dim
        self.num_relations = num_relations

    def forward(self, sbjs, objs, is_train=True, only_means=True):

        if is_train and only_means:
            return F.bilinear(sbjs, objs, self.bilinear_means), torch.norm(self.bilinear_means,p=self.p, dim=(1,2))
        elif not is_train:
            return F.bilinear(sbjs, objs, self.bilinear_means)
        else:
            rels_noise = self.N.sample(self.bilinear_means.shape).to(sbjs.device)
            entropic_loss = torch.sum(self.bilinear_logstd)
            rels = self.bilinear_means + torch.exp(self.bilinear_logstd) * rels_noise
            reg_loss_rels = torch.norm(rels,p=self.p, dim=(1,2))

            return F.bilinear(sbjs, objs, rels), reg_loss_rels, entropic_loss


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

        self.mus = torch.randn(num_relations, emb_dim * 2)
        self.mus.requires_grad = True
        self.sigmas = torch.randn(num_relations, emb_dim * 2).abs() + 1.
        self.sigmas.requires_grad = True

        self.sbj_normals = [
            [
                Normal(self.mus[rel, sbj], self.sigmas[rel, sbj])
                for sbj in range(emb_dim)
            ]
            for rel in range(num_relations)
        ]

        self.obj_normals = [
            [
                Normal(self.mus[rel, obj], self.sigmas[rel, obj])
                for obj in range(emb_dim, emb_dim*2)
            ]
            for rel in range(num_relations)
        ]

        self.emb_dim = emb_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.relation_priors = relation_priors

    def forward(self, sbjs, objs):
        assert sbjs.shape[1:] == (self.emb_dim,)
        assert objs.shape[1:] == (self.emb_dim,)

        probs = []
        for rel in range(self.num_relations):
            rel_probs = []
            for sbj in range(self.emb_dim):
                sbj_normal = self.sbj_normals[rel][sbj]
                rel_probs.append(sbj_normal.log_prob(sbjs[:, sbj]))

            for obj in range(self.emb_dim):
                obj_normal = self.obj_normals[rel][obj]
                rel_probs.append(obj_normal.log_prob(objs[:, obj]))

            probs.append(torch.stack(rel_probs))
        probs = torch.stack(probs).permute(2, 0, 1)
        assert probs.shape[1:] == (self.num_relations, self.emb_dim*2,)

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
    e_counts = e_counts.float().to(device)
    r_counts = torch.bincount(torch.Tensor(train_rels).int(),minlength=num_relations).float().to(device)
    ents_lambdas = initial_l2_reg * e_counts
    rels_lambdas = initial_l2_reg * r_counts

    train_set = MyDataset(train_sbjs, train_objs, train_rels)
    dev_set   = MyDataset(dev_sbjs,   dev_objs,   dev_rels)
    test_set  = MyDataset(test_sbjs,  test_objs,  test_rels)

    BATCH_SIZE = 32
    # BATCH_SIZE = 64
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

    # def onepass_variational(embs, model, optim, is_train, loader, ents_lambdas, rels_lambdas, phase, lambdas_lr=0.8, p=2):
    def onepass_variational(embs, model, optim, is_train, loader, ents_lambdas, rels_lambdas, phase, lambdas_lr=0.97, p=2):
        """

        embs :: VariationalNormalEmbs
        p :: 2, order of norm (keep at 2 for now)
        ents_lambdas :: size: num_entities, specifies regularization per entity
        phase :: one of 'pre/retrain', 'E', or 'EM'; explained below
        lambdas_lr :: parameter between (0,1) specifying how fast to update lambdas (1 corresponds to no averaging)

        In the variational EM paper, training is divided into four phases:
        (1) Pre-training: train only embedding means with fixed regularization
        (2) Pure E step: train embedding means + embedding logstd with fixed regularization
        (3) EM Step: train embedding means + embedding logstd while updating regularization
        (4) Re-training: same as (1) but using updated regularization terms

        """

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

                if is_train and phase=='pre/retrain':
                    sbjs_embs, sbjs_reg_loss = embs(sbjs, is_train=True, only_means=True)
                    objs_embs, objs_reg_loss = embs(objs, is_train=True, only_means=True)
                    logits, rels_reg_loss = model(sbjs_embs, objs_embs, is_train=True, only_means=True)
                elif is_train and (phase=='E' or phase=='EM'):
                    sbjs_embs, sbjs_reg_loss, sbjs_entropic_loss = embs(sbjs, is_train=True, only_means=False)
                    objs_embs, objs_reg_loss, objs_entropic_loss = embs(objs, is_train=True, only_means=False)
                    logits, rels_reg_loss, rels_entropic_loss = model(sbjs_embs, objs_embs, is_train=True, only_means=False)
                elif not is_train:
                    sbjs_embs = embs(sbjs, is_train=False)
                    objs_embs = embs(objs, is_train=False)
                    logits = model(sbjs_embs, objs_embs, is_train=False)


                loss = F.cross_entropy(logits, rels, reduction="none")
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)

                if is_train:

                    train_loss = torch.sum(loss)

                    sbjs_lambdas = torch.gather(ents_lambdas,0,sbjs)
                    objs_lambdas = torch.gather(ents_lambdas,0,objs)

                    batch_sbj_counts = torch.bincount(sbjs.int(), minlength=embs.num_entities).float()
                    batch_obj_counts = torch.bincount(objs.int(), minlength=embs.num_entities).float()
                    batch_subj_norm = torch.gather(batch_sbj_counts,0,sbjs)
                    batch_obj_norm = torch.gather(batch_obj_counts,0,objs)

                    sbjs_lambdas = sbjs_lambdas/batch_subj_norm
                    objs_lambdas = objs_lambdas/batch_obj_norm

                    loss +=  torch.dot(sbjs_lambdas,sbjs_reg_loss) / p
                    loss +=  torch.dot(objs_lambdas,objs_reg_loss) / p
                    loss +=  torch.dot(rels_lambdas,rels_reg_loss) / p

                    if (phase=='E' or phase=='EM'):
                        loss -=  sbjs_entropic_loss
                        loss -=  objs_entropic_loss
                        loss -=  rels_entropic_loss

                    train_loss.backward()

                batch_losses.append(loss.cpu())
                batch_preds.append(preds.cpu())
                batch_rels.append(rels.cpu())

                if is_train:
                    optim.step()

                if is_train and phase=='EM':

                    ents_lambdas_update = torch.square(embs.embs_means.weight.clone().detach())
                    ents_lambdas_update += torch.square(torch.exp(embs.embs_logstd.weight.clone().detach()))
                    ents_lambdas_update = torch.sum(ents_lambdas_update, dim=1)
                    ents_lambdas = ((1. - lambdas_lr)/ents_lambdas) + ((lambdas_lr / embs.emb_dim) * ents_lambdas_update)
                    ents_lambdas = 1. / ents_lambdas

                    rels_lambdas_update = torch.square(model.bilinear_means.clone().detach())
                    rels_lambdas_update += torch.square(torch.exp(model.bilinear_logstd.clone().detach()))
                    rels_lambdas_update = torch.sum(rels_lambdas_update, dim=(1,2))
                    rels_lambdas = ((1. - lambdas_lr)/rels_lambdas) + ((lambdas_lr / model.emb_dim**2) * rels_lambdas_update)
                    rels_lambdas = 1. / rels_lambdas

        loss = float(torch.cat(batch_losses).mean())
        preds = torch.cat(batch_preds)
        rels = torch.cat(batch_rels)

        acc = float((preds == rels).to(float).mean())
        f1 = f1_score(rels, preds, average="macro")

        if is_train and phase=='EM':
            return acc, f1, loss, ents_lambdas

        return (acc, f1, loss)

    # Normal embs

    emb_dim = 300

    # normal_embs = NormalEmbs(num_entities, emb_dim, 0., 0.001)
    # normal_embs.to(device)

    var_embs = VariationalEntityEmbs(num_entities, emb_dim, 0., 0.001)
    var_embs.to(device)

    # normal_bilinear = Bilinear(emb_dim, num_relations)
    # normal_bilinear.to(device)
    # priors = torch.zeros(num_relations).to(device)
    # normal_nb = NaiveBayes(emb_dim, num_entities, num_relations, priors)
    # normal_nb.to(device)

    var_bilinear = VariationalBilinear(num_relations=num_relations, emb_dim=emb_dim, mean=0., var=0.001)
    var_bilinear.to(device)


    # normal_optim = optim.Adam(list(normal_embs.parameters()) + list(normal_bilinear.parameters()))
    # normal_optim = optim.Adam(list(normal_embs.parameters()) + list(normal_nb.parameters()))
    var_optim = optim.Adam(list(var_embs.parameters()) + list(var_bilinear.parameters()))

    for phase in phases:
        print(f"Beginning {phase} phase")
        training_metrics = []

        print("Normal embeddings...")

        for epoch in range(num_epochs):
            # train_acc, train_f1, train_loss = onepass(normal_embs, normal_bilinear, normal_optim, True,  train_loader)
            # dev_acc,   dev_f1,   dev_loss   = onepass(normal_embs, normal_bilinear, None,         False, dev_loader)
            # train_acc, train_f1, train_loss = onepass(normal_embs, normal_nb, normal_optim, True,  train_loader)
            # dev_acc,   dev_f1,   dev_loss   = onepass(normal_embs, normal_nb, None,         False, dev_loader)

            if phase=='EM':
                train_acc, train_f1, train_loss, ents_lambdas = onepass_variational(var_embs, var_bilinear, var_optim, is_train=True, loader=train_loader,
                                                                        # ents_lambdas=ents_lambdas, rels_lambdas=rels_lambdas, phase=phase, lambdas_lr=0.8, p=2)
                                                                        ents_lambdas=ents_lambdas, rels_lambdas=rels_lambdas, phase=phase, lambdas_lr=0.97, p=2)
            else:
                train_acc, train_f1, train_loss = onepass_variational(var_embs, var_bilinear, var_optim, is_train=True, loader=train_loader,
                                                                        # ents_lambdas=ents_lambdas, rels_lambdas=rels_lambdas, phase=phase, lambdas_lr=0.8, p=2)
                                                                        ents_lambdas=ents_lambdas, rels_lambdas=rels_lambdas, phase=phase, lambdas_lr=0.97, p=2)

            dev_acc,   dev_f1,   dev_loss   = onepass_variational(var_embs, var_bilinear, None, is_train=False, loader=dev_loader,
                                                                    # ents_lambdas=ents_lambdas, rels_lambdas=rels_lambdas, phase=None, lambdas_lr=0.8, p=2)
                                                                    ents_lambdas=ents_lambdas, rels_lambdas=rels_lambdas, phase=None, lambdas_lr=0.97, p=2)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"\tTraining accuracy: {train_acc:.3f},\tF1: {train_f1:.3f},\tLoss: {float(train_loss):.3f}")
            print(f"\tValidation accuracy: {dev_acc:.3f},\tF1: {dev_f1:.3f},\tLoss: {float(dev_loss):.3f}")

            training_metrics.append(( epoch+1, train_acc, train_f1, train_loss, "train", "normal", dataname ))
            training_metrics.append(( epoch+1, dev_acc, dev_f1, dev_loss, "dev", "normal", dataname ))

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



NUM_EPOCHS = 10

import csv

def update_csv(path, rows):
    with open(path, "a") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

# RESULTS_PATH = "results-bilinear.csv"
# RESULTS_PATH = "results-nb.csv"
RESULTS_PATH = "results-var-bilinear.csv"

# update_csv(
#     RESULTS_PATH,
#     [ ( "epoch", "acc", "f1", "loss", "set", "emb", "data" ) ]
# )

# print("Training wn18rr...", file=sys.stderr)
# update_csv(
#     RESULTS_PATH,
#     setup(wn18rr_raw,    "wn18rr",    NUM_EPOCHS)
# )

# print("Training on wn11...", file=sys.stderr)
# update_csv(
#     RESULTS_PATH,
#     setup(wn11_raw,      "wn11",      NUM_EPOCHS)
# )

print("Training on fb15k-237...", file=sys.stderr)
update_csv(
    RESULTS_PATH,
    setup(fb15k_237_raw, "fb15k-237", NUM_EPOCHS)
)

print("Training on fb13...", file=sys.stderr)
update_csv(
    RESULTS_PATH,
    setup(fb13_raw,      "fb13",      NUM_EPOCHS)
)

print("Training on yago3-10...", file=sys.stderr)
update_csv(
    RESULTS_PATH,
    setup(yago3_10_raw,  "yago3-10",  NUM_EPOCHS)
)