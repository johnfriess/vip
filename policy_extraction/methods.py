import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_value_model(snapshot_path, device="cuda"):
    """Load a frozen value model from a training snapshot.

    Expects checkpoint format: {'model': state_dict, 'model_cfg': {...}, ...}
    where model_cfg contains '_target_' and constructor kwargs.
    """
    payload = torch.load(snapshot_path, map_location="cpu")
    cfg = payload["model_cfg"]
    target = cfg["_target_"]
    model_kwargs = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "device"}

    if "StateVIP" in target:
        from vip.models.model_vip import StateVIP
        model = StateVIP(**model_kwargs)
    elif "VIP" in target:
        from vip.models.model_vip import VIP
        model = VIP(**model_kwargs)
    else:
        raise ValueError(f"Unknown model target: {target}")

    state_dict = {k.replace("module.", ""): v for k, v in payload["model"].items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    is_image = "State" not in target
    return model, is_image


class BaseTrainer:
    """Shared logic for all policy extraction methods.

    Subclasses set two flags that control data flow:
      use_pretrained  - True  → image policy uses its own CNN (CNNGaussianPolicy)
                           False → policy receives pre-computed VIP embeddings
      uses_embeddings    - True  → state policy receives VIP embeddings
                           False → state policy receives raw states

    Subclasses override _compute_weights / _epoch_weight to change how
    demonstration transitions are weighted during training.
    """

    label = "Base"

    def __init__(self, snapshot_path, device):
        model, is_image = load_value_model(snapshot_path, device)
        self.model = model
        self.is_image = is_image
        self.hidden_dim = model.hidden_dim
        self.device = device
        # Subclasses override these in their __init__
        self.use_pretrained = False
        self.uses_embeddings = False

    # -- hooks for subclasses --------------------------------------------------

    def _compute_weights(self, emb_s, emb_sn, emb_g):
        """Return per-sample weights. Base: uniform (all ones)."""
        return torch.ones(len(emb_s))

    def _epoch_weight(self, w, epoch, epochs):
        """Adjust per-batch weights at the start of each epoch. Base: identity."""
        return w

    # -- shared implementation -------------------------------------------------

    def _precompute(self, dataset, batch_size):
        """Pre-embed data through frozen VIP. Returns (train_s, actions, train_g, emb_s, emb_sn, emb_g)."""
        device = self.device
        model = self.model

        if self.is_image:
            print(f"  Pre-embedding {len(dataset)} image transitions...")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)
            all_emb_s, all_a, all_emb_sn, all_emb_g = [], [], [], []
            all_img_s, all_img_g = [], []
            with torch.no_grad():
                for batch in loader:
                    img_s, a, img_sn, img_g = batch
                    combined = torch.cat([img_s, img_sn, img_g], dim=0).to(device)
                    emb = model(combined).cpu()
                    bs = len(img_s)
                    all_emb_s.append(emb[:bs])
                    all_a.append(a)
                    all_emb_sn.append(emb[bs:2*bs])
                    all_emb_g.append(emb[2*bs:])
                    if not self.uses_embeddings:
                        all_img_s.append(img_s)
                        all_img_g.append(img_g)
            emb_s  = torch.cat(all_emb_s)
            actions = torch.cat(all_a)
            emb_sn = torch.cat(all_emb_sn)
            emb_g  = torch.cat(all_emb_g)
            if self.uses_embeddings:
                train_s = emb_s
                train_g = emb_g
            else:
                train_s = torch.cat(all_img_s)
                train_g = torch.cat(all_img_g)
        else:
            s_np  = torch.from_numpy(dataset.states)
            a_np  = torch.from_numpy(dataset.actions)
            sn_np = torch.from_numpy(dataset.next_states)
            g_np  = torch.from_numpy(dataset.goals)
            all_emb_s, all_emb_sn, all_emb_g = [], [], []
            with torch.no_grad():
                for i in range(0, len(s_np), batch_size):
                    all_emb_s.append(model(s_np[i:i+batch_size].to(device)).cpu())
                    all_emb_sn.append(model(sn_np[i:i+batch_size].to(device)).cpu())
                    all_emb_g.append(model(g_np[i:i+batch_size].to(device)).cpu())
            emb_s  = torch.cat(all_emb_s)
            emb_sn = torch.cat(all_emb_sn)
            emb_g  = torch.cat(all_emb_g)
            actions = a_np
            if self.uses_embeddings:
                train_s = emb_s
                train_g = emb_g
            else:
                train_s = s_np
                train_g = g_np

        return train_s, actions, train_g, emb_s, emb_sn, emb_g

    def train(self, dataset, policy, epochs, batch_size, lr):
        device = self.device

        train_s, actions, train_g, emb_s, emb_sn, emb_g = self._precompute(dataset, batch_size)
        weights = self._compute_weights(emb_s, emb_sn, emb_g)
        train_loader = DataLoader(
            TensorDataset(train_s, actions, train_g, weights),
            batch_size=batch_size, shuffle=True,
        )

        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for s, a, g, w in train_loader:
                s, a, g, w = s.to(device), a.to(device), g.to(device), w.to(device)
                per_sample = F.mse_loss(policy.get_action_mean(s, g), a, reduction='none').mean(-1)
                w = self._epoch_weight(w, epoch, epochs)
                loss = (w * per_sample).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(train_loader)
            losses.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [{self.label}] epoch {epoch+1}/{epochs}  loss={avg:.5f}")
        return losses


class BCTrainer(BaseTrainer):
    """Plain behavior cloning — uniform-weighted MSE.

    Image models: trainable CNN encoder (CNNGaussianPolicy with its own ResNet).
    State models: policy receives raw states.
    """

    label = "BC"

    def __init__(self, snapshot_path, device):
        super().__init__(snapshot_path, device)
        self.use_pretrained = self.is_image
        self.uses_embeddings = False


class RepTransferTrainer(BaseTrainer):
    """Behavior cloning in VIP embedding space — policy receives frozen VIP
    embeddings as input instead of raw observations. No trainable encoder."""

    label = "RepTransfer"

    def __init__(self, snapshot_path, device):
        super().__init__(snapshot_path, device)
        self.use_pretrained = False
        self.uses_embeddings = True


class AWRTrainer(BaseTrainer):
    """Advantage-weighted regression from a frozen VIP value model.

    Image models: trainable CNN encoder (CNNGaussianPolicy with its own ResNet).
    State models: policy receives raw states.
    """

    label = "AWR"

    def __init__(self, snapshot_path, temperature, max_weight, device):
        super().__init__(snapshot_path, device)
        self.temperature = temperature
        self.max_weight = max_weight
        self.use_pretrained = self.is_image
        self.uses_embeddings = False

    def _compute_weights(self, emb_s, emb_sn, emb_g):
        with torch.no_grad():
            adv = (self.model.sim(emb_sn.to(self.device), emb_g.to(self.device))
                   - self.model.sim(emb_s.to(self.device), emb_g.to(self.device)))
        weights = torch.exp(adv.cpu() / self.temperature).clamp(max=self.max_weight)
        weights = weights / (weights.mean() + 1e-8)
        return weights


class GradualAWRTrainer(AWRTrainer):
    """BC-first, gradually value-guided policy extraction.

    L_t = (1 - alpha_e) * L_BC + alpha_e * L_AWR
    where alpha_e = (epoch / (epochs - 1)) ^ warmup_exponent.

    warmup_exponent controls the transition shape:
        > 1: stay with BC longer, ramp AWR late  (e.g. 2.0 = quadratic warmup)
        = 1: linear transition
        < 1: ramp AWR early, less BC emphasis     (e.g. 0.5 = sqrt warmup)
    """

    label = "GradualAWR"

    def __init__(self, snapshot_path, temperature, max_weight, warmup_exponent, device):
        super().__init__(snapshot_path, temperature, max_weight, device)
        self.warmup_exponent = warmup_exponent

    def _epoch_weight(self, w, epoch, epochs):
        alpha = (epoch / max(epochs - 1, 1)) ** self.warmup_exponent
        return (1.0 - alpha) + alpha * w
