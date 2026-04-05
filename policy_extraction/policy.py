import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


class GaussianMLPPolicy(nn.Module):
    """Goal-conditioned Gaussian policy: pi(a | s, g).

    Input:  state and goal tensors (concatenated internally)
    Output: Gaussian distribution over actions
    """

    def __init__(self, input_dim, action_dim, hidden_dims=(256, 256),
                 log_std_init=-0.5, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_action_mean(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        h = self.trunk(x)
        return self.mean_head(h)

    @torch.no_grad()
    def act(self, state, goal, deterministic=False):
        dist = self.forward(state, goal)
        action = dist.mean if deterministic else dist.sample()
        return action.cpu().numpy()


class CNNGaussianPolicy(nn.Module):
    """Goal-conditioned Gaussian policy with trainable CNN encoder.

    Each frame in the stack is encoded independently through a shared
    pretrained ResNet (standard 3-channel input, no conv1 modification).
    Per-frame embeddings are concatenated, then fed through an MLP trunk.
    """

    def __init__(self, action_dim, frame_stack=1, encoder_size=18,
                 encoder_dim=256, hidden_dims=(256, 256),
                 log_std_init=-0.5, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.action_dim = action_dim
        self.frame_stack = frame_stack
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared ResNet encoder (ImageNet-pretrained, standard 3-channel input)
        if encoder_size == 18:
            self.encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            outdim = 512
        elif encoder_size == 50:
            self.encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            outdim = 2048
        else:
            raise ValueError(f"Unsupported encoder_size: {encoder_size}")

        self.encoder.fc = nn.Linear(outdim, encoder_dim)

        # Freeze all conv layers, only train fc + MLP head
        for name, param in self.encoder.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad_(False)

        # Standard ImageNet preprocessing (always 3-channel)
        self.preprocess = nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )

        # MLP trunk: input = (state frames + goal frames) * encoder_dim
        mlp_input_dim = 2 * frame_stack * encoder_dim
        layers = [nn.LayerNorm(mlp_input_dim), nn.Linear(mlp_input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def encode(self, img):
        """Encode stacked frames independently. img: [B, 3*frame_stack, H, W] float [0-255]."""
        x = img.float() / 255.0 if img.max() > 2.0 else img
        B = x.shape[0]
        # Split into individual 3-channel frames and encode each
        frames = x.unfold(1, 3, 3).permute(0, 1, 4, 2, 3)  # [B, frame_stack, 3, H, W]
        frames = frames.reshape(B * self.frame_stack, 3, x.shape[2], x.shape[3])  # [B*fs, 3, H, W]
        frames = self.preprocess(frames)
        embs = self.encoder(frames)  # [B*fs, encoder_dim]
        return embs.reshape(B, -1)  # [B, frame_stack * encoder_dim]

    def forward(self, img_s, img_g):
        emb_s = self.encode(img_s)
        emb_g = self.encode(img_g)
        x = torch.cat([emb_s, emb_g], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_action_mean(self, img_s, img_g):
        emb_s = self.encode(img_s)
        emb_g = self.encode(img_g)
        x = torch.cat([emb_s, emb_g], dim=-1)
        return self.mean_head(self.trunk(x))

    @torch.no_grad()
    def act(self, img_s, img_g, deterministic=False):
        dist = self.forward(img_s, img_g)
        action = dist.mean if deterministic else dist.sample()
        return action.cpu().numpy()
