import torch
import torch.nn as nn
import torch.nn.functional as F

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        m = 0.5 * (p + q)
        jsd = 0.5 * (self.kl_div(F.log_softmax(p, dim=1), m) + self.kl_div(F.log_softmax(q, dim=1), m))
        return jsd