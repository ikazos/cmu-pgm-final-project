import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import contextlib
from sklearn.metrics import f1_score
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalNormalEmbs(nn.Module):
    def __init__(self, num_entities, emb_dim, mean, var):
        """
        Same as Normal Embs class except each embedding component has a mean and variance.
        """
        super(VariationalNormalEmbs, self).__init__()

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
            return ents_means, torch.norm(ents_means,p=self.p, dim=1)
        elif not is_train:
            return ents_means
        else:
            ents_logstd = self.embs_logstd(ents)
            # entropic_loss = torch.sum(ents_logstd) #return sum of log standard deviations; this is the entropy term of a gaussian
            entropic_loss = torch.sum(self.embs_logstd.weight) #add loss for ALL logstddevs instead of those in the batch
            ents_logstd = torch.exp(ents_logstd)

            ents_noise = self.N.sample(ents_logstd.shape) #not actually noise, reparametrization trick
            ents = ents_means + ents_logstd * ents_noise
            reg_loss_ents = torch.norm(ents,p=self.p, dim=1) #need to return norm of embeddings with noise to compute L2 loss

            return ents, reg_loss_ents, entropic_loss



def onepass_variational(embs, model, optim, is_train, loader, ents_lambdas, phase, lambdas_lr=0.8, p=2):
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
            elif is_train and (phase=='E' or phase=='EM'):
                sbjs_embs, sbjs_reg_loss, sbjs_entropic_loss = embs(sbjs, is_train=True, only_means=False)
                objs_embs, objs_reg_loss, objs_entropic_loss = embs(objs, is_train=True, only_means=False)
            elif not is_train:
                sbjs_embs = embs(sbjs, is_train=False)
                objs_embs = embs(objs, is_train=False)

            logits = model(sbjs_embs, objs_embs)
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

                if (phase=='E' or phase=='EM'):
                    loss -=  sbjs_entropic_loss
                    loss -=  objs_entropic_loss

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

    loss = float(torch.cat(batch_losses).mean())
    preds = torch.cat(batch_preds)
    rels = torch.cat(batch_rels)

    acc = float((preds == rels).to(float).mean())
    f1 = f1_score(rels, preds, average="macro")

    if is_train and phase=='EM':
        return acc, f1, loss, ents_lambdas

    return (acc, f1, loss)
