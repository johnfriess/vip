# import torch
# import torch.nn as nn
# from utils import mlp, weight_init
# from torchvision import transforms
# import torch.nn.functional as F
# import torchvision

# class TwinQ(nn.Module):
#     def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         dims = [3*state_dim, *([hidden_dim] * n_hidden), 1]
#         self.q1 = mlp(dims, squeeze_output=True)
#         print("Q1")
#         print(self.q1)
#         self.q2 = mlp(dims, squeeze_output=True)

#     def both(self, state, next_state, goal):
#         sa = torch.cat([state, next_state, goal], 1)
#         return self.q1(sa), self.q2(sa)

#     def forward(self, state, next_state, goal):
#         return torch.min(*self.both(state, next_state, goal))
    


# class ValueFunction(nn.Module):
#     def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         dims = [2*state_dim, *([hidden_dim] * n_hidden), 1]
#         self.v = mlp(dims, squeeze_output=True)

#     def forward(self, state, goal):
#         return self.v(torch.cat((state, goal), dim=1))
    
# class StateEncoder(nn.Module):
#     def __init__(self, state_dim, hidden_dims=[256, 256], output_dim=2):
#         super().__init__()
#         dims = [state_dim, *hidden_dims, output_dim]
#         self.phi = mlp(dims, squeeze_output=False)

#     def forward(self, state):
#         return self.phi(state.float())
    
# class RandomShiftsAug(nn.Module):
#     def __init__(self, pad):
#         super().__init__()
#         self.pad = pad

#     def forward(self, x):
#         n, c, h, w = x.size()
#         assert h == w
#         padding = tuple([self.pad] * 4)
#         x = F.pad(x, padding, 'replicate')
#         eps = 1.0 / (h + 2 * self.pad)
#         arange = torch.linspace(-1.0 + eps,
#                                 1.0 - eps,
#                                 h + 2 * self.pad,
#                                 device=x.device,
#                                 dtype=x.dtype)[:h]
#         arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
#         base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
#         base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

#         shift = torch.randint(0,
#                               2 * self.pad + 1,
#                               size=(n, 1, 1, 2),
#                               device=x.device,
#                               dtype=x.dtype)
#         shift *= 2.0 / (h + 2 * self.pad)

#         grid = base_grid + shift
#         return F.grid_sample(x,
#                              grid,
#                              padding_mode='zeros',
#                              align_corners=False)
    
# class DrQEncoder(nn.Module):
#     def __init__(self, obs_shape):
#         super().__init__()

#         assert len(obs_shape) == 3
#         self.repr_dim = 32 * 35 * 35

#         self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
#                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                      nn.ReLU())

#         self.apply(weight_init)

#     def forward(self, obs):
#         obs = obs / 255.0 - 0.5
#         h = self.convnet(obs)
#         h = h.view(h.shape[0], -1)
#         return h
    
# class ImageEncoder(nn.Module):
#     def __init__(self, arch='', hidden_dims=[256, 256]):
#         super().__init__()
#         self.normlayer = self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#         if arch != 'drq':
#             self.preprocess = nn.Sequential(
#                             transforms.Resize(256),
#                             transforms.CenterCrop(224),
#                             self.normlayer,
#             )
#         else:
#             self.preprocess = nn.Sequential(
#                             transforms.Resize(84),
#                             RandomShiftsAug(pad=4),
#             )

#         if arch == 'drq':
#             self.phi = DrQEncoder((3, 84, 84))
#             self.repr_dim = self.phi.repr_dim
#         elif arch == 'resnet50':
#             self.phi = torchvision.models.resnet50(pretrained=False)
#             self.phi.fc = nn.Identity()
#             self.repr_dim = 2048
#         elif arch == 'vit':
#             from transformers import AutoConfig
#             self.phi = AutoModel.from_config(config = AutoConfig.from_pretrained('google/vit-base-patch32-224-in21k'))

#             self.repr_dim = 768
#         if len(hidden_dims) > 0:
#             dims = [self.repr_dim, *hidden_dims]
#             self.phi = nn.Sequential(self.preprocess, self.phi, mlp(dims, squeeze_output=False))

#     def forward(self, obs):
#         return self.phi(obs.float())



# class EncoderICVFValueFunction(nn.Module):
#     def __init__(self, state_dim, hidden_dims=[256, 256,2], vf_type='l2'):
#         super().__init__()
#         assert vf_type in ['l2', 'dot', 'mlp','multilinear']
#         self.vf_type = vf_type
#         dims = [state_dim, *hidden_dims]
#         self.phi = mlp(dims, squeeze_output=False)
#         self.psi = mlp(dims, squeeze_output=False)
        
#         # self.phi_inverse = mlp([*reversed(hidden_dims),state_dim], squeeze_output=False)

#         if self.vf_type == 'multilinear':
#             self.T = mlp([hidden_dims[-1], *hidden_dims], squeeze_output=False)
#             self.matrix_a = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
#             self.matrix_b = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
#         elif self.vf_type == 'mlp':
#             self.v = mlp([3*hidden_dims[-1], 1], squeeze_output=True)
#             # self.v = mlp([2*hidden_dims[-1], 256, 1], squeeze_output=True)

#     def encode(self, state):
#         state = state.float()
#         return self.phi(state)
    
#     def decode(self, enc):
#         state = self.phi_inverse(enc.float())
#         return state



#     def sim(self, phi_s,psi_fs, psi_g):
#         # if self.vf_type == 'l2':
#         #     return -torch.linalg.norm(phi_s - phi_g, dim=-1)
#         # elif self.vf_type == 'dot':
#         #     return -torch.sum(phi_s * phi_g, dim=1)
#         # else:
#         if self.vf_type == 'multilinear':
#             Tz = self.T(psi_g)
#             phi_z = self.matrix_a(phi_s*Tz)
#             psi_z = self.matrix_b(psi_fs*Tz)
#             return (phi_z * psi_z).sum(axis=-1)
#         else:
#             return self.v(torch.cat((phi_s,psi_fs, psi_g), dim=1))
        
#     def forward(self, state, future_state, goal):
#         phi_s = self.phi(state)
#         psi_fs = self.psi(future_state)
        
#         phi_g = self.psi(goal)
#         return self.sim(phi_s,psi_fs, phi_g)
    


# class EncoderValueFunction(nn.Module):
#     def __init__(self, arch, state_dim, hidden_dims=[256, 256,2], vf_type='l2'):
#         super().__init__()
#         assert vf_type in ['l2', 'dot', 'mlp','multilinear']
#         self.vf_type = vf_type
#         dims = [state_dim, *hidden_dims]
#         if arch=='mlp':
#             self.phi = mlp(dims, squeeze_output=False)
#             self.psi = mlp(dims, squeeze_output=False)
#             self.phi_inverse = mlp([*reversed(hidden_dims),state_dim], squeeze_output=False)
#         else:
#             self.phi = ImageEncoder(arch=arch, hidden_dims=hidden_dims)
#         # self.phi_inverse = mlp([*reversed(hidden_dims),state_dim], squeeze_output=False)
#         print("V")
#         print(self.phi)
#         if self.vf_type == 'multilinear':
#             self.T = mlp([hidden_dims[-1], *hidden_dims], squeeze_output=False)
#             self.matrix_a = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
#             self.matrix_b = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
#         elif self.vf_type == 'mlp':
#             self.v = mlp([2*hidden_dims[-1], 1], squeeze_output=True)
#             # self.v = mlp([2*hidden_dims[-1], 256, 1], squeeze_output=True)

#     def encode(self, state):
#         state = state.float()
#         return self.phi(state)
    
#     def decode(self, enc):
#         state = self.phi_inverse(enc.float())
#         return state

#     def sim(self, phi_s, psi_g):
#         if self.vf_type == 'l2':
#             return -torch.linalg.norm(phi_s - psi_g, dim=-1)
#         elif self.vf_type == 'dot':
#             return -torch.sum(phi_s * psi_g, dim=1)
#         elif self.vf_type == 'multilinear':
#             Tz = self.T(psi_g)
#             phi_z = self.matrix_a(phi_s*Tz)
#             psi_z = self.matrix_b(psi_g*Tz)
#             return (phi_z * psi_z).sum(axis=-1)
#         else:
#             return self.v(torch.cat((phi_s, psi_g), dim=1))
        
#     def forward(self, state, goal):
#         phi_s = self.phi(state)
#         phi_g = self.psi(goal)
#         return self.sim(phi_s, phi_g)
    
# class L2ValueFunction(nn.Module):
#     def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         dims = [state_dim, *([hidden_dim] * n_hidden), 16]
#         self.phi = mlp(dims, squeeze_output=False)

#     def forward(self, state, goal):
#         return torch.sum(self.phi(state) * self.phi(goal), 1)


import torch
import torch.nn as nn
from utils import mlp, weight_init
from torchvision import transforms
import torch.nn.functional as F
import torchvision

class TwinQ(nn.Module):
    def __init__(self, arch, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.arch = arch
        dims = [3*state_dim, *([hidden_dim] * n_hidden), 1]
        if arch=='mlp':
            self.q1 = mlp(dims, squeeze_output=True)
            self.q2 = mlp(dims, squeeze_output=True)
        else:
            self.q1 = ImageEncoder(arch=arch, hidden_dims=[])
            self.q2 = ImageEncoder(arch=arch, hidden_dims=[])

            dims = [3 * self.q1.repr_dim, *([hidden_dim] * n_hidden), 1]
            print(f"MLP input dimensions: {dims}")

            self.out1 = mlp(dims, squeeze_output=True)
            self.out2 = mlp(dims, squeeze_output=True)

        print("Q1")
        print(self.q1)
        

    def both(self, state, next_state, goal):
        # print(f"State shape {state.shape}")
        # print(f"next state shape  {next_state.shape}")
        # print(f"goal shape (g1 {goal.shape}")

        if self.arch != 'mlp':
            e_state1, e_next_state1, e_goal1 = self.q1(state), self.q1(next_state), self.q1(goal)
            e_state2, e_next_state2, e_goal2 = self.q2(state), self.q2(next_state), self.q2(goal)

            sa1 = torch.cat([e_state1, e_next_state1, e_goal1], 1)
            sa2 = torch.cat([e_state2, e_next_state2, e_goal2], 1)

            return self.out1(sa1), self.out2(sa2)
        else:
            sa = torch.cat([state, next_state, goal], 1)
            return self.q1(sa), self.q2(sa)

    def forward(self, state, next_state, goal):
        return torch.min(*self.both(state, next_state, goal))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [2*state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state, goal):
        return self.v(torch.cat((state, goal), dim=1))
    

class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256], output_dim=2):
        super().__init__()
        dims = [state_dim, *hidden_dims, output_dim]
        self.phi = mlp(dims, squeeze_output=False)

    def forward(self, state):
        return self.phi(state.float())
    
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
class DrQEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    
class ImageEncoder(nn.Module):
    def __init__(self, arch='', hidden_dims=[256, 256]):
        super().__init__()
        self.normlayer = self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if arch != 'drq':
            self.preprocess = nn.Sequential(
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            self.normlayer,
            )
        else:
            self.preprocess = nn.Sequential(
                            transforms.Resize(84),
                            RandomShiftsAug(pad=4),
            )

        if arch == 'drq':
            self.phi = DrQEncoder((3, 84, 84))
            self.repr_dim = self.phi.repr_dim
        elif arch == 'resnet50':
            self.phi = torchvision.models.resnet34(pretrained=True)
            # requires grad = False
            for param in self.phi.parameters():
                param.requires_grad = False
            self.phi.fc = nn.Identity()
            # self.repr_dim = 2048
            self.repr_dim = 512 * 4
        elif arch == 'vit':
            from transformers import AutoConfig
            self.phi = AutoModel.from_config(config = AutoConfig.from_pretrained('google/vit-base-patch32-224-in21k'))

            self.repr_dim = 768
        if len(hidden_dims) > 0:
            dims = [self.repr_dim, *hidden_dims]
            # self.phi = nn.Sequential(self.preprocess, self.phi, mlp(dims, squeeze_output=False))
            self.phi = nn.Sequential(self.preprocess, self.phi)
            self.output_phi = mlp(dims, squeeze_output=False)

    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        if len(obs.shape) == 5:
            # extract frames
            num_frames = obs.shape[1]
            # obs = [obs[:, i*3:(i+1)*3] for i in range(num_frames)]
            obs = [obs[:, i] for i in range(num_frames)]

            # process each frame
            obs = [self.phi(frame) for frame in obs]
            obs = torch.cat(obs, dim=1)
        else:
            obs = self.phi(obs.float())
        return self.output_phi(obs)
        # return self.phi(obs.float())

class EncoderICVFValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256,2], vf_type='l2'):
        super().__init__()
        assert vf_type in ['l2', 'dot', 'mlp','multilinear']
        self.vf_type = vf_type
        dims = [state_dim, *hidden_dims, 2]
        self.phi = mlp(dims, squeeze_output=False)
        self.psi = mlp(dims, squeeze_output=False)
        
        self.phi_inverse = mlp([*reversed(hidden_dims),state_dim], squeeze_output=False)

        if self.vf_type == 'multilinear':
            self.T = mlp([hidden_dims[-1], *hidden_dims], squeeze_output=False)
            self.matrix_a = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
            self.matrix_b = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
        elif self.vf_type == 'mlp':
            self.v = mlp([3*hidden_dims[-1], 1], squeeze_output=True)
            # self.v = mlp([2*hidden_dims[-1], 256, 1], squeeze_output=True)

    def encode(self, state):
        state = state.float()
        return self.phi(state)
    
    def decode(self, enc):
        state = self.phi_inverse(enc.float())
        return state



    def sim(self, phi_s,psi_fs, psi_g):
        # if self.vf_type == 'l2':
        #     return -torch.linalg.norm(phi_s - phi_g, dim=-1)
        # elif self.vf_type == 'dot':
        #     return -torch.sum(phi_s * phi_g, dim=1)
        # else:
        if self.vf_type == 'multilinear':
            Tz = self.T(psi_g)
            phi_z = self.matrix_a(phi_s*Tz)
            psi_z = self.matrix_b(psi_fs*Tz)
            return (phi_z * psi_z).sum(axis=-1)
        else:
            return self.v(torch.cat((phi_s,psi_fs, psi_g), dim=1))
        
    def forward(self, state, future_state, goal):
        phi_s = self.phi(state)
        psi_fs = self.psi(future_state)
        
        phi_g = self.psi(goal)
        return self.sim(phi_s,psi_fs, phi_g)

class EncoderValueFunction(nn.Module):
    def __init__(self, arch, state_dim, hidden_dims=[256, 256,2], vf_type='l2'):
        super().__init__()
        assert vf_type in ['l2', 'dot', 'mlp','multilinear']
        self.vf_type = vf_type
        dims = [state_dim, *hidden_dims]
        if arch=='mlp':
            self.phi = mlp(dims, squeeze_output=False)
            self.psi = mlp(dims, squeeze_output=False)
            self.phi_inverse = mlp([*reversed(hidden_dims),state_dim], squeeze_output=False)
        else:
            self.phi = ImageEncoder(arch=arch, hidden_dims=hidden_dims)
            self.psi = ImageEncoder(arch=arch, hidden_dims=hidden_dims)
        print("V")
        print(self.phi)
        if self.vf_type == 'multilinear':
            self.T = mlp([hidden_dims[-1], *hidden_dims], squeeze_output=False)
            self.matrix_a = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
            self.matrix_b = mlp([hidden_dims[-1], hidden_dims[-1]], squeeze_output=False)
        elif self.vf_type == 'mlp':
            self.v = mlp([2*hidden_dims[-1], 1], squeeze_output=True)
            # self.v = mlp([2*hidden_dims[-1], 256, 1], squeeze_output=True)

    def encode(self, state):
        state = state.float()
        return self.phi(state)
    
    def decode(self, enc):
        state = self.phi_inverse(enc.float())
        return state

    def sim(self, phi_s, psi_g):
        if self.vf_type == 'l2':
            return -torch.linalg.norm(phi_s - psi_g, dim=-1)
        elif self.vf_type == 'dot':
            return -torch.sum(phi_s * psi_g, dim=1)
        elif self.vf_type == 'multilinear':
            Tz = self.T(psi_g)
            phi_z = self.matrix_a(phi_s*Tz)
            psi_z = self.matrix_b(psi_g*Tz)
            return (phi_z * psi_z).sum(axis=-1)
        else:
            return self.v(torch.cat((phi_s, psi_g), dim=1))
        
    def forward(self, state, goal):
        phi_s = self.phi(state)
        psi_g = self.psi(goal)
        return self.sim(phi_s, psi_g)
    
class L2ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 16]
        self.phi = mlp(dims, squeeze_output=False)

    def forward(self, state, goal):
        return torch.sum(self.phi(state) * self.phi(goal), 1)
    