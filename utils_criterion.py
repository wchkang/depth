import torch
import torch.nn as nn
import torch.nn.functional as F

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
      
    
    def forward(self, logits_p, logits_q, eps=1e-8):
        p = F.softmax(logits_p, dim=1).clamp(min=eps)
        q = F.softmax(logits_q, dim=1).clamp(min=eps)

        m = 0.5 * (p + q)
        log_m = torch.log(m)

        kl_pm = F.kl_div(log_m, p, reduction='batchmean')
        kl_qm = F.kl_div(log_m, q, reduction='batchmean')

        jsd = 0.5 * (kl_pm + kl_qm)

        return jsd
        