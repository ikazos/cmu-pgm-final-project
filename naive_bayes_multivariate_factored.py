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


class MultivariateFactoredNB(nn.Module):
    def __init__(self, emb_dim, basis_dim, num_entities,
                       num_relations, relation_priors, train_prior=False):
        super(MultivariateFactoredNB, self).__init__()

        self.emb_dim = emb_dim
        self.basis_dim = basis_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        assert (relation_priors>0).all().item()
        self.register_parameter("log_priors", Parameter(torch.log(relation_priors/torch.sum(relation_priors))))
        self.log_priors.requires_grad = train_prior

        self.register_parameter("sbjs_mus", Parameter(torch.randn(num_relations, emb_dim)))
        self.register_parameter("objs_mus", Parameter(torch.randn(num_relations, emb_dim)))

        with torch.no_grad():
            mask = 1 - torch.diag(torch.ones(emb_dim))
            self.register_parameter("chols", Parameter(mask * nn.init.normal_(torch.empty((basis_dim, emb_dim, emb_dim)))))

        self.register_parameter("sbjs_diags", nn.init.normal_(Parameter(torch.empty((basis_dim, emb_dim)))))
        self.register_parameter("objs_diags", nn.init.normal_(Parameter(torch.empty((basis_dim, emb_dim)))))

        # self.register_parameter("rels_coeff", nn.init.normal_(Parameter(torch.empty((basis_dim,num_relations)))))


    def forward(self, sbjs, objs):

        sbjs = sbjs[:,None,:] - self.sbjs_mus[None,:,:]
        objs = objs[:,None,:] - self.objs_mus[None,:,:]

        diag_s = torch.exp(self.sbjs_diags)
        diag_o = torch.exp(self.objs_diags)

        L_sbjs = torch.tril(self.chols,-1) + torch.diag_embed(diag_s)
        L_objs = torch.triu(self.chols, 1) + torch.diag_embed(diag_o)

        sbjs_left = torch.einsum('brn,knm->brkm',sbjs,L_sbjs)
        sbjs = torch.einsum('brkm,brkm->bk',sbjs_left,sbjs_left)
        # sbjs = torch.einsum('bk,kr->br',sbjs,self.rels_coeff)

        objs_right = torch.einsum('knm,brm->brkn',L_objs,objs)
        objs = torch.einsum('brkn,brkn->bk',objs_right,objs_right)
        # objs = torch.einsum('bk,kr->br',objs,self.rels_coeff)

        logits = sbjs + objs
        logits *= -1./2.
        logits = logits + self.log_priors[None,:]

        return logits
