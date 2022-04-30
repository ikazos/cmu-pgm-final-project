import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal


class NormalEmbs(nn.Module):
    def __init__(self, num_entities, emb_dim, mean, var):
        super(NormalEmbs, self).__init__()

        n = Normal(mean, var)
        weights = n.sample((num_entities, emb_dim))
        self.embs = nn.Embedding.from_pretrained(weights, freeze=False)

    def forward(self, ents):
        return self.embs(ents)


class DimensionCorrelatedNB(nn.Module):
    def __init__(self, emb_dim, num_entities,
                       num_relations, relation_priors, train_prior=False):
        super(DimensionCorrelatedNB, self).__init__()

        self.emb_dim = emb_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        assert (relation_priors>0).all().item()
        self.log_priors = Parameter(torch.log(relation_priors/torch.sum(relation_priors)))
        self.log_priors.requires_grad = train_prior

        self.sbjs_mus = Parameter(torch.randn(num_relations, emb_dim))
        self.objs_mus = Parameter(torch.randn(num_relations, emb_dim))

        self.sbjs_logstds = Parameter(torch.randn(num_relations, emb_dim))
        self.objs_logstds = Parameter(torch.randn(num_relations, emb_dim))

        self.subj_objs_corrs = Parameter(torch.randn(num_relations, emb_dim))


    def forward(self, sbjs, objs):

        sbjs = sbjs[:,None,:] - self.sbjs_mus[None,:,:]
        objs = objs[:,None,:] - self.objs_mus[None,:,:]

        sbjs_stds = torch.exp(self.sbjs_logstds)
        objs_stds = torch.exp(self.objs_logstds)

        sbjs = sbjs/sbjs_stds[None,:,:]
        objs = objs/objs_stds[None,:,:]

        corrs = torch.tanh(self.subj_objs_corrs)

        logits = sbjs*sbjs + objs*objs - 2*corrs[None,:,:]*sbjs*objs
        logits = (-1. / 2.) * logits / (1. - (corrs**2)[None,:,:])
        logits = torch.sum(logits,dim=2)
        logits = logits + self.log_priors[None,:]

        return logits
