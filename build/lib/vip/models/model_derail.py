import torch
import torch.nn as nn

def mlp(state_dim, size, mlp_width, output_dim):
    layers = [nn.LayerNorm(state_dim), nn.Linear(state_dim, mlp_width), nn.ReLU()]
    for _ in range(size):
            layers.append(nn.Linear(mlp_width, mlp_width))
            layers.append(nn.ReLU())
    layers.append(nn.Linear(mlp_width, output_dim))
    return nn.Sequential(*layers)

class StateEuclideanDERAIL(nn.Module):
    def __init__(self, device="cuda", lr=1e-4, state_dim=60, hidden_dim=256, size=1, mlp_width=256, l2weight=1.0, l1weight=1.0, conservative_weight=0.9, gamma=0.98, num_negatives=0):
        super().__init__()
        self.device = device
        self.l2weight = l2weight
        self.l1weight = l1weight

        self.hidden_dim = hidden_dim
        self.conservative_weight = conservative_weight
        self.gamma = gamma
        self.size = size
        self.num_negatives = num_negatives

        self.encoder = mlp(state_dim, size, mlp_width, hidden_dim)

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(self.parameters(), lr = lr)

    ## Forward Call (state --> representation)
    def forward(self, obs):
        h = self.encoder(obs)
        return h

    def sim(self, tensor1, tensor2):
        d = -torch.linalg.norm(tensor1 - tensor2, dim = -1)
        return d

class StateMultiLinearDERAIL(nn.Module):
    def __init__(self, device="cuda", lr=1e-4, state_dim=60, hidden_dim=256, size=1, mlp_width=256, l2weight=1.0, l1weight=1.0, conservative_weight=0.9, gamma=0.98, num_negatives=0):
        super().__init__()
        self.device = device
        self.l2weight = l2weight
        self.l1weight = l1weight

        self.hidden_dim = hidden_dim
        self.conservative_weight = conservative_weight
        self.gamma = gamma
        self.size = size
        self.num_negatives = num_negatives

        ## phi1(s): (B, h)
        self.encoder1 = mlp(state_dim, size, mlp_width, hidden_dim) 

        ## goal trunk: (B, width)
        self.goal_trunk = mlp(state_dim, size, mlp_width, mlp_width)

        ## two heads from goal trunk
        ## phi2(g): (B, h*h)
        ## phi3(g): (B, h)
        self.encoder2 = nn.Linear(mlp_width, hidden_dim * hidden_dim)
        self.encoder3 = nn.Linear(mlp_width, hidden_dim)

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(self.parameters(), lr = lr)

    ## Forward Call (state, goal --> representations)
    def forward(self, s, g):
        h1 = self.encoder1(s)
        trunk = self.goal_trunk(g)
        h2 = self.encoder2(trunk).view(-1, self.hidden_dim, self.hidden_dim)
        h3 = self.encoder3(trunk)
        return h1, h2, h3

    def value(self, tensor1, tensor2, tensor3):
        val = tensor1.unsqueeze(1) @ tensor2 @ tensor3.unsqueeze(-1)
        return val.view(-1)
    