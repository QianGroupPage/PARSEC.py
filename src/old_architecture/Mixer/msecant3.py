import numpy as np
from .qinv import qinv
from .includemix import get_params


class mixer:
    """
    Hybrid multi-secant update.
    """
    def __init__(self):
        self.params = get_params()
        self.reset_state()

    def reset_state(self):
        self.DX = None
        self.DF = None
        self.N = None
        self.DX1 = None
        self.DX2 = None
        self.DF2 = None
        self.EN_stage = None
        self.type = None

    def mixer(self, x1, f1):
        x_new, m = self.msecant3(x1, f1)
        return x_new, m

    def _ensure_buffers(self, n, sz):
        if self.N is None or self.N.shape[1] != sz:
            self.N = np.zeros((n, sz))
        if self.DX1 is None or self.DX1.shape[1] != sz:
            self.DX1 = np.zeros((n, sz))
        if self.DX2 is None or self.DX2.shape[1] != sz:
            self.DX2 = np.zeros((n, sz))
        if self.DF2 is None or self.DF2.shape[1] != sz:
            self.DF2 = np.zeros((n, sz))

    def _init_first(self, x1, f1, EN_like):
        self.DX = x1[:, np.newaxis]
        self.DF = f1[:, np.newaxis]
        self.EN_stage = 0 if EN_like == 0 else 2
        self.type = self.params["preferred_type"]
        return x1 + self.params["mix"] * f1, 0

    def msecant3(self, x1, f1):
        p = self.params
        mix = p["mix"]
        group_size = p["group_size"]
        tol = p["tol"]
        restart_factor = p["restart_factor"]
        EN_like = p["EN_like"]
        preferred_type = p["preferred_type"]

        if self.DX is None:
            return self._init_first(x1, f1, EN_like)

        m = self.DX.shape[1]
        sz = m + 1 if group_size == 0 else group_size
        self._ensure_buffers(x1.size, sz)

        if self.EN_stage != 1 and m >= 2 and np.linalg.norm(self.DF[:, -1], 2) < restart_factor * np.linalg.norm(f1, 2):
            self.reset_state()
            self.params = p
            return self._init_first(x1, f1, EN_like)

        res = (m + sz - 1) % sz + 1
        ngroup = (m - res) // sz

        if self.EN_stage != 1:
            self.DX[:, m - 1] = x1 - self.DX[:, m - 1]
            self.DF[:, m - 1] = f1 - self.DF[:, m - 1]
            self.DX1[:, res - 1] = self.DX[:, m - 1]
            self.DX[:, m - 1] += mix * self.DF[:, m - 1]
            for i in range(ngroup):
                self.DX[:, m - 1] -= self.DX[:, i * sz:(i + 1) * sz] @ (
                    self.DF[:, i * sz:(i + 1) * sz].T @ self.DF[:, m - 1]
                )

        x_new = x1 + mix * f1
        for i in range(ngroup):
            x_new -= self.DX[:, i * sz:(i + 1) * sz] @ (self.DF[:, i * sz:(i + 1) * sz].T @ f1)

        if self.EN_stage == 1 and sz == res:
            x_new -= self.DX[:, ngroup * sz:m] @ (self.DF[:, ngroup * sz:m].T @ f1)
            self.DX = np.hstack([self.DX, x1[:, np.newaxis]])
            self.DF = np.hstack([self.DF, f1[:, np.newaxis]])
            self.EN_stage = 2
            return x_new, m

        if self.EN_stage != 1:
            self.N[:, res - 1] = -mix * self.DX1[:, res - 1]
            for i in range(ngroup):
                self.N[:, res - 1] += self.DF[:, i * sz:(i + 1) * sz] @ (
                    self.DX[:, i * sz:(i + 1) * sz].T @ self.DX1[:, res - 1]
                )

        if self.EN_stage != 1 and ngroup > 0:
            lhs = np.linalg.norm(self.DF[:, m - res:m].T @ self.DF2[:, sz - res:sz], ord="fro") * \
                  np.linalg.norm(self.N[:, :res].T @ self.DF[:, m - res:m], ord="fro")
            rhs = np.linalg.norm(self.DF[:, m - res:m].T @ self.DF[:, m - res:m], ord="fro") * \
                  np.linalg.norm(self.DX1[:, :res].T @ self.DX2[:, sz - res:sz], ord="fro")
            if lhs > rhs:
                self.type = 1
            elif lhs < rhs:
                self.type = 2
            else:
                self.type = preferred_type
        else:
            self.type = preferred_type

        if self.type == 1:
            M = self.N[:, :res].T @ self.DF[:, m - res:m]
            C = qinv(M, tol)
            if res == sz:
                self.DX2[:, :sz] = self.DX1
                self.DF2[:, :sz] = self.DF[:, m - res:m]
                self.DF[:, m - res:m] = self.N[:, :res] @ C.T
                x_new -= self.DX[:, m - res:m] @ (self.DF[:, m - res:m].T @ f1)
            else:
                x_new -= self.DX[:, m - res:m] @ ((C @ self.N[:, :res].T) @ f1)
        else:
            C2 = qinv(self.DF[:, m - res:m], tol)
            C = C2 @ C2.T
            if res == sz:
                self.DX2[:, :sz] = self.DX1
                self.DF2[:, :sz] = self.DF[:, m - res:m]
                self.DF[:, m - res:m] = self.DF[:, m - res:m] @ C
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
