"""DPO objective."""

import copy

import torch
import torch.nn.functional as F


def sequence_logprob(model, input_ids, labels):
    out = model(input_ids, labels=None)["logits"]
    logp = F.log_softmax(out[:, :-1], dim=-1)
    target = labels[:, 1:].unsqueeze(-1)
    return logp.gather(-1, target).squeeze(-1).sum(dim=-1)


class DPOTrainer:
    def __init__(self, model, beta: float):
        self.model = model
        self.ref_model = copy.deepcopy(model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.beta = beta

    def loss(self, chosen_ids, rejected_ids):
        pi_c = sequence_logprob(self.model, chosen_ids, chosen_ids)
        pi_r = sequence_logprob(self.model, rejected_ids, rejected_ids)
        with torch.no_grad():
            ref_c = sequence_logprob(self.ref_model, chosen_ids, chosen_ids)
            ref_r = sequence_logprob(self.ref_model, rejected_ids, rejected_ids)
        logits = self.beta * ((pi_c - pi_r) - (ref_c - ref_r))
        return -torch.logsigmoid(logits).mean()
