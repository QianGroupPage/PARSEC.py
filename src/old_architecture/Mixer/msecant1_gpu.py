import cupy as cp

from .includemix import get_params
from .qinv_gpu import qinv


class mixer:
    """
    GPU multi-secant Type-I update with persistent device-resident history.
    """

    def __init__(self):
        self.params = get_params()
        self.reset_state()

    def reset_state(self):
        """Reset persistent variables."""
        self.DX = None
        self.DF = None
        self.N = None
        self.EN_stage = None

    def mixer(self, x1, f1):
        """Main entry point."""
        x_new, m = self.msecant1(x1, f1)
        return x_new, m

    def msecant1(self, x1, f1):
        """Core msecant1 algorithm on the GPU."""
        p = self.params
        mix = p["mix"]
        group_size = p["group_size"]
        tol = p["tol"]
        restart_factor = p["restart_factor"]
        EN_like = p["EN_like"]

        x1 = cp.asarray(x1).reshape(-1)
        f1 = cp.asarray(f1).reshape(-1)

        if self.DX is None:
            m = 0
        else:
            m = self.DX.shape[1]

        if m == 0:
            self.DX = x1[:, cp.newaxis].copy()
            self.DF = f1[:, cp.newaxis].copy()

            x_new = x1 + mix * f1

            if EN_like == 0:
                self.EN_stage = 0
            else:
                self.EN_stage = 2

            return x_new, m

        if group_size == 0:
            sz = m + 1
        else:
            sz = group_size

        if self.EN_stage != 1 and m >= 2:
            norm_df_last = cp.linalg.norm(self.DF[:, m - 1], 2)
            norm_f1 = cp.linalg.norm(f1, 2)
            if float(norm_df_last.item()) < restart_factor * float(norm_f1.item()):
                x1_restart = self.DX[:, m - 1].copy()
                f1_restart = self.DF[:, m - 1].copy()

                self.DX = x1_restart[:, cp.newaxis]
                self.DF = f1_restart[:, cp.newaxis]
                self.N = None

                x_new = x1_restart + mix * f1_restart
                return x_new, 0

        res = (m + sz - 1) % sz + 1
        ngroup = (m - res) // sz

        if self.EN_stage != 1:
            self.DX[:, m - 1] = x1 - self.DX[:, m - 1]
            self.DF[:, m - 1] = f1 - self.DF[:, m - 1]

            dx = self.DX[:, m - 1].copy()

            self.DX[:, m - 1] = self.DX[:, m - 1] + mix * self.DF[:, m - 1]

            for i in range(1, ngroup + 1):
                start_idx = (i - 1) * sz
                end_idx = i * sz
                self.DX[:, m - 1] = self.DX[:, m - 1] - (
                    self.DX[:, start_idx:end_idx]
                    @ (self.DF[:, start_idx:end_idx].T @ self.DF[:, m - 1])
                )
        else:
            dx = None

        x_new = x1 + mix * f1

        for i in range(1, ngroup + 1):
            start_idx = (i - 1) * sz
            end_idx = i * sz
            x_new = x_new - self.DX[:, start_idx:end_idx] @ (
                self.DF[:, start_idx:end_idx].T @ f1
            )

        if self.EN_stage != 1:
            n_size = x1.size
            if self.N is None or self.N.shape[1] < res:
                self.N = cp.zeros((n_size, sz), dtype=x1.dtype)

            self.N[:, res - 1] = -mix * dx

            for i in range(1, ngroup + 1):
                start_idx = (i - 1) * sz
                end_idx = i * sz
                self.N[:, res - 1] = self.N[:, res - 1] + (
                    self.DF[:, start_idx:end_idx]
                    @ (self.DX[:, start_idx:end_idx].T @ dx)
                )

        if self.EN_stage == 1 and res == sz:
            start_idx = ngroup * sz
            end_idx = m
            if end_idx > start_idx:
                x_new = x_new - self.DX[:, start_idx:end_idx] @ (
                    self.DF[:, start_idx:end_idx].T @ f1
                )
        else:
            M = self.N[:, :res].T @ self.DF[:, m - res:m]
            C = qinv(M, tol)

            if res == sz:
                self.DF[:, m - sz:m] = self.N[:, :sz] @ C.T
                x_new = x_new - self.DX[:, m - res:m] @ (
                    self.DF[:, m - res:m].T @ f1
                )
            else:
                x_new = x_new - self.DX[:, m - res:m] @ (
                    (C @ self.N[:, :res].T) @ f1
                )

        if self.EN_stage != 2:
            self.DX = cp.hstack([self.DX, x1[:, cp.newaxis]])
            self.DF = cp.hstack([self.DF, f1[:, cp.newaxis]])

        if self.EN_stage == 1:
            self.EN_stage = 2
        elif self.EN_stage == 2:
            self.EN_stage = 1

        return x_new, m


_persistent_mixer = None


def get_mixer(reset=False):
    """Get or create the persistent GPU mixer instance."""
    global _persistent_mixer
    if reset or _persistent_mixer is None:
        _persistent_mixer = mixer()
    return _persistent_mixer


def reset_mixer():
    """Reset the persistent GPU mixer state."""
    global _persistent_mixer
    _persistent_mixer = None


def mixer_step(x1, f1, reset=False):
    """Convenience function."""
    return get_mixer(reset=reset).mixer(x1, f1)
