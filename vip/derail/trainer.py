import time

class Trainer():
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, log=None, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()

        ## Batch
        obs_0, goals, obs, next_obs, gt_s0, gt_s1, rewards, terminals = batch

        v_loss, q_loss = model.update(obs.float().cuda(), next_obs.float().cuda(), goals.float().cuda(),gt_s0.float().cuda(),gt_s1.float().cuda(), rewards.float().cuda(), terminals.float().cuda(), log)

        metrics['v_loss'] = v_loss
        metrics['q_loss'] = q_loss

        t2 = time.time()

        return metrics, f"Load time {t1-t0}, Batch time {t2-t1}, Update time {t2-t1}, Q Loss {metrics['q_loss']}, V Loss {metrics['v_loss']}"
