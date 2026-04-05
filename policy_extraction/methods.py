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


class AWRTrainer:
    """Advantage-weighted regression from a frozen VIP value model."""

    def __init__(self, snapshot_path, temperature, max_weight, device):
        model, is_image = load_value_model(snapshot_path, device)
        self.model = model
        self.is_image = is_image
        self.hidden_dim = model.hidden_dim
        self.temperature = temperature
        self.max_weight = max_weight
        self.device = device

    def train(self, dataset, policy, epochs, batch_size, lr, trainable_encoder=False):
        device = self.device
        model = self.model

        if self.is_image:
            # Pre-embed all images through frozen VIP for advantage computation
            print(f"  Pre-embedding {len(dataset)} image transitions...")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)
            all_emb_s, all_a, all_emb_sn, all_emb_g = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    img_s, a, img_sn, img_g = batch[:4]
                    combined = torch.cat([img_s, img_sn, img_g], dim=0).to(device)
                    emb = model(combined).cpu()
                    bs = len(img_s)
                    all_emb_s.append(emb[:bs])
                    all_a.append(a)
                    all_emb_sn.append(emb[bs:2*bs])
                    all_emb_g.append(emb[2*bs:])
            emb_s  = torch.cat(all_emb_s)
            actions = torch.cat(all_a)
            emb_sn = torch.cat(all_emb_sn)
            emb_g  = torch.cat(all_emb_g)
            train_s = emb_s
            train_g = emb_g
        else:
            # State: encode for advantages, train policy on raw states
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
            train_s = s_np
            train_g = g_np

        # Compute per-sample advantage weights
        with torch.no_grad():
            adv = model.sim(emb_sn.to(device), emb_g.to(device)) - model.sim(emb_s.to(device), emb_g.to(device))
        weights = torch.exp(adv.cpu() / self.temperature).clamp(max=self.max_weight)
        weights = weights / (weights.mean() + 1e-8)

        if trainable_encoder and self.is_image:
            # Attach weights to dataset so DataLoader returns them
            dataset.weights = weights
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True, persistent_workers=True)
        else:
            train_loader = DataLoader(
                TensorDataset(train_s, actions, train_g, weights),
                batch_size=batch_size, shuffle=True,
            )

        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                if trainable_encoder and self.is_image:
                    img_s, a, _, img_g, w = batch
                    img_s, a, img_g, w = img_s.to(device), a.to(device), img_g.to(device), w.to(device)
                    per_sample = F.mse_loss(policy.get_action_mean(img_s, img_g), a, reduction='none').mean(-1)
                else:
                    s, a, g, w = batch
                    s, a, g, w = s.to(device), a.to(device), g.to(device), w.to(device)
                    per_sample = F.mse_loss(policy.get_action_mean(s, g), a, reduction='none').mean(-1)
                loss = (w * per_sample).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(train_loader)
            losses.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [AWR] epoch {epoch+1}/{epochs}  loss={avg:.5f}")
        return losses


class GradualAWRTrainer:
    """BC-first, gradually value-guided policy extraction.

    L_t = (1 - alpha_e) * L_BC + alpha_e * L_AWR
    where alpha_e = (epoch / (epochs - 1)) ^ warmup_exponent.

    warmup_exponent controls the transition shape:
        > 1: stay with BC longer, ramp AWR late  (e.g. 2.0 = quadratic warmup)
        = 1: linear transition
        < 1: ramp AWR early, less BC emphasis     (e.g. 0.5 = sqrt warmup)
    """

    def __init__(self, snapshot_path, temperature, max_weight, warmup_exponent, device):
        model, is_image = load_value_model(snapshot_path, device)
        self.model = model
        self.is_image = is_image
        self.hidden_dim = model.hidden_dim
        self.temperature = temperature
        self.max_weight = max_weight
        self.warmup_exponent = warmup_exponent
        self.device = device

    def train(self, dataset, policy, epochs, batch_size, lr, trainable_encoder=False):
        device = self.device
        model = self.model

        if self.is_image:
            print(f"  Pre-embedding {len(dataset)} image transitions...")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)
            all_emb_s, all_a, all_emb_sn, all_emb_g = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    img_s, a, img_sn, img_g = batch[:4]
                    combined = torch.cat([img_s, img_sn, img_g], dim=0).to(device)
                    emb = model(combined).cpu()
                    bs = len(img_s)
                    all_emb_s.append(emb[:bs])
                    all_a.append(a)
                    all_emb_sn.append(emb[bs:2*bs])
                    all_emb_g.append(emb[2*bs:])
            emb_s  = torch.cat(all_emb_s)
            actions = torch.cat(all_a)
            emb_sn = torch.cat(all_emb_sn)
            emb_g  = torch.cat(all_emb_g)
            train_s = emb_s
            train_g = emb_g
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
            train_s = s_np
            train_g = g_np

        # Compute per-sample advantage weights
        with torch.no_grad():
            adv = model.sim(emb_sn.to(device), emb_g.to(device)) - model.sim(emb_s.to(device), emb_g.to(device))
        weights = torch.exp(adv.cpu() / self.temperature).clamp(max=self.max_weight)
        weights = weights / (weights.mean() + 1e-8)

        if trainable_encoder and self.is_image:
            dataset.weights = weights
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True, persistent_workers=True)
        else:
            train_loader = DataLoader(
                TensorDataset(train_s, actions, train_g, weights),
                batch_size=batch_size, shuffle=True,
            )

        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            alpha = (epoch / max(epochs - 1, 1)) ** self.warmup_exponent
            epoch_loss = 0.0
            for batch in train_loader:
                if trainable_encoder and self.is_image:
                    img_s, a, _, img_g, w = batch
                    img_s, a, img_g, w = img_s.to(device), a.to(device), img_g.to(device), w.to(device)
                    per_sample = F.mse_loss(policy.get_action_mean(img_s, img_g), a, reduction='none').mean(-1)
                else:
                    s, a, g, w = batch
                    s, a, g, w = s.to(device), a.to(device), g.to(device), w.to(device)
                    per_sample = F.mse_loss(policy.get_action_mean(s, g), a, reduction='none').mean(-1)
                effective_w = (1.0 - alpha) + alpha * w
                loss = (effective_w * per_sample).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(train_loader)
            losses.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [GradualAWR] epoch {epoch+1}/{epochs}  loss={avg:.5f}  alpha={alpha:.3f}")
        return losses
