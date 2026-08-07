import numpy as np
from .qinv import qinv
from .includemix import get_params


class mixer:
    """
    Multi-secant Type-II update.
    """
    def __init__(self):
        self.params = get_params()
        self.reset_state()

    def reset_state(self):
        self.DX = None
        self.DF = None
        self.EN_stage = None

    def mixer(self, x1, f1):
        x_new, m = self.msecant2(x1, f1)
        return x_new, m

    def msecant2(self, x1, f1):
        p = self.params
        mix = p["mix"]
        group_size = p["group_size"]
        tol = p["tol"]
        restart_factor = p["restart_factor"]
        EN_like = p["EN_like"]

        if self.DX is None:
            m = 0
            self.DX = x1[:, np.newaxis]
            self.DF = f1[:, np.newaxis]
            x_new = x1 + mix * f1
            self.EN_stage = 0 if EN_like == 0 else 2
            return x_new, m

        m = self.DX.shape[1]
        sz = m + 1 if group_size == 0 else group_size

        if self.EN_stage != 1 and m >= 2 and np.linalg.norm(self.DF[:, -1], 2) < restart_factor * np.linalg.norm(f1, 2):
            self.DX = x1[:, np.newaxis]
            self.DF = f1[:, np.newaxis]
            x_new = x1 + mix * f1
            self.EN_stage = 2
            m = 0
            return x_new, m

        res = (m + sz - 1) % sz + 1
        ngroup = (m - res) // sz

        if self.EN_stage != 1:
            self.DX[:, m - 1] = x1 - self.DX[:, m - 1]
            self.DF[:, m - 1] = f1 - self.DF[:, m - 1]
            self.DX[:, m - 1] += mix * self.DF[:, m - 1]
            for i in range(ngroup):
                self.DX[:, m - 1] -= self.DX[:, i * sz:(i + 1) * sz] @ (
                    self.DF[:, i * sz:(i + 1) * sz].T @ self.DF[:, m - 1]
                )

        x_new = x1 + mix * f1
        for i in range(ngroup):
            x_new -= self.DX[:, i * sz:(i + 1) * sz] @ (self.DF[:, i * sz:(i + 1) * sz].T @ f1)

        if self.EN_stage == 1 and res == sz:
            x_new -= self.DX[:, ngroup * sz:m] @ (self.DF[:, ngroup * sz:m].T @ f1)
        else:
            C2 = qinv(self.DF[:, m - res:m], tol)
            C = C2 @ C2.T
            if res == sz:
                self.DF[:, m - sz:m] = self.DF[:, m - sz:m] @ C
                x_new -= self.DX[:, m - res:m] @ (self.DF[:, m - res:m].T @ f1)
            else:
                x_new -= self.DX[:, m - res:m] @ ((self.DF[:, m - res:m] @ C).T @ f1)

        if self.EN_stage != 2:
            self.DX = np.hstack([self.DX, x1[:, np.newaxis]])
            self.DF = np.hstack([self.DF, f1[:, np.newaxis]])

        if self.EN_stage == 1:
            self.EN_stage = 2
        elif self.EN_stage == 2:
            self.EN_stage = 1

        return x_new, m


_persistent_mixer = None


def get_mixer(reset=False):
    global _persistent_mixer
    if reset or _persistent_mixer is None:
        _persistent_mixer = mixer()
    return _persistent_mixer


def mixer_step(x1, f1, reset=False):
    return get_mixer(reset=reset).mixer(x1, f1)
