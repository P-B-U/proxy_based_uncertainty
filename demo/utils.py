import torch
import math

# density of sigmoid transformed gaussian distribution
def f(x, m=0, s=1, a=0.00001, b=0.99999, n=10000):
    return (1/s) * (1/(x*(1-x))) * torch.exp(-((torch.log(x/(1-x))-m)/s)**2 / 2) / (math.sqrt(2 * math.pi))

# posterior density of f
def fplus(x, ef_val, m=0, s=1, a=0.00001, b=0.99999, n=10000):
    return x*f(x, m, s, a, b, n) / ef_val.reshape(ef_val.shape[0], 1)

# numerical integration, use Trapezoidal method
def tintf(m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    y = f(x, m, s)
    result = (h / 2) * (2 * torch.sum(y, dim = 1) - y[:, 0] - y[:, n])
    return result

# expectation of f
def ef(m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    fv = f(x, m, s)
    idx = fv <= 1e-10

    rem = 1 - tintf(m, s, a, b, n)
    leftover = rem * n
    idx = rem <= 0
    leftover[idx] = 0

    y = x* fv
    result = (h / 2) * (2 * torch.sum(y, dim = 1) - y[:, 0] - y[:, n]) + leftover
    return result

# differential entropy of posterior f
def hfplus(ef_val, m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    fv = fplus(x, ef_val, m, s)
    idx = fv <= 1e-10
    logfv = torch.log(fv)
    logfv[idx] = 0

    rem = 1 - tintf(m, s, a, b, n)
    leftover = -rem * n * torch.log(rem * n)
    idx = rem <= 0
    leftover[idx] = 0

    y = -fv * logfv
    result = (h / 2) * (2 * torch.sum(y, dim = 1) - y[:, 0] - y[:, n]) + leftover
    return result

# Shannon entropy of Ef
def hef(m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    fv = f(x, m, s)
    
    y = fv * x
    p = (h / 2) * (2 * torch.sum(y, dim=1) - y[:, 0] - y[:, n])

    p[p>b] = b
    p[p<a] = a
    
    result = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    return result

def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits), dim=dim, keepdim=keepdim)

def logit_mean(logits, dim: int, keepdim: bool = False):
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])

def balanced_entropy(m, s):
    ef_val = ef(m,s)
    bal_ent = (ef_val*hfplus(ef_val, m,s) + (1-ef_val)*hfplus(1-ef_val, -m,s) + hef(m,s))/(hef(m,s)+0.6931472)
    return bal_ent

def epistemic_uncertainty(logits):
    sample_entropies = entropy(logits, dim=-1)
    entropy_mean = torch.mean(sample_entropies, dim=1)

    logits_mean = logit_mean(logits, dim=1)
    mean_entropy = entropy(logits_mean, dim=-1)

    mutual_info = mean_entropy - entropy_mean

    idx = mutual_info < 1e-9
    mutual_info[idx] = mutual_info[idx] * 0 + 1e-9

    return mutual_info

def aleatoric_uncertainty(logits):
    pred_entropy = entropy(logit_mean(logits, dim=1, keepdim=False), dim=-1)
    epistemic_unc = epistemic_uncertainty(logits)

    return pred_entropy - epistemic_unc