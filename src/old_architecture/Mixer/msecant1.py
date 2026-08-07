import numpy as np
from .qinv import qinv
from .includemix import get_params


class mixer:
    """
    Multi-secant Type-I update
    
    Multi-secant methods for solving nonlinear equations f(x)=0;
    Type-I methods that minimize the change of approximate Jacobian.
    
    Mixing parameters are defined in includemix.py (get_params()).
    
    Input:
        x1 = latest iterate;
        f1 = f(x1) = potNew - pot (the residual).
    
    Output:
        x_new = new estimate of the solution to f(x)=0;
        m = number of secant equations.
    """
    
    def __init__(self):
        self.params = get_params()
        self.reset_state()

    def reset_state(self):
        """Reset persistent variables."""
        self.DX = None  # Will store x differences and E vectors
        self.DF = None  # Will store f differences and V vectors
        self.N = None   # Stores the last group N_i
        self.EN_stage = None

    def mixer(self, x1, f1):
        """Main entry point."""
        x_new, m = self.msecant1(x1, f1)
        return x_new, m

    def msecant1(self, x1, f1):
        """
        Core msecant1 algorithm
        """
        # Get parameters from includemix
        p = self.params
        mix = p["mix"]
        group_size = p["group_size"]
        tol = p["tol"]
        restart_factor = p["restart_factor"]
        EN_like = p["EN_like"]

        # Ensure x1 and f1 are 1D arrays
        x1 = np.asarray(x1).flatten()
        f1 = np.asarray(f1).flatten()

        # m = number of previous iterates (columns in DX)
        if self.DX is None:
            m = 0
        else:
            m = self.DX.shape[1]

        # First iteration - no previous iterate available
        if m == 0:
            # Store the current iterate
            self.DX = x1[:, np.newaxis].copy()
            self.DF = f1[:, np.newaxis].copy()
            
            # Simple mixing: x_new = x1 + mix*f1
            x_new = x1 + mix * f1
            
            # Set EN_stage based on EN_like
            if EN_like == 0:
                self.EN_stage = 0  # Broyden-like update
            else:
                self.EN_stage = 2  # EN-like update, next stage is 2
            
            return x_new, m

        # Determine group size
        if group_size == 0:
            sz = m + 1  # Take all available iterates in one group
        else:
            sz = group_size

        # Check for restart condition
        if self.EN_stage != 1 and m >= 2:
            norm_df_last = np.linalg.norm(self.DF[:, m-1], 2)
            norm_f1 = np.linalg.norm(f1, 2)
            if norm_df_last < restart_factor * norm_f1:
                # Perform restart
                x1_restart = self.DX[:, m-1].copy()
                f1_restart = self.DF[:, m-1].copy()
                
                self.DX = x1_restart[:, np.newaxis]
                self.DF = f1_restart[:, np.newaxis]
                self.N = None
                
                x_new = x1_restart + mix * f1_restart
                return x_new, 0

        # Compute res (size of last group) and ngroup
        res = (m + sz - 1) % sz + 1
        ngroup = (m - res) // sz

        # Update DX and DF if not in EN_stage 1
        if self.EN_stage != 1:
            self.DX[:, m-1] = x1 - self.DX[:, m-1]  # dx_m = x_{m+1} - x_m
            self.DF[:, m-1] = f1 - self.DF[:, m-1]  # df_m = f_{m+1} - f_m
            
            dx = self.DX[:, m-1].copy()
            
            self.DX[:, m-1] = self.DX[:, m-1] + mix * self.DF[:, m-1]
            
            for i in range(1, ngroup + 1):
                start_idx = (i - 1) * sz
                end_idx = i * sz
                self.DX[:, m-1] = self.DX[:, m-1] - \
                    self.DX[:, start_idx:end_idx] @ (self.DF[:, start_idx:end_idx].T @ self.DF[:, m-1])
        else:
            dx = None

        # Compute new x
        x_new = x1 + mix * f1
        
        for i in range(1, ngroup + 1):
            start_idx = (i - 1) * sz
            end_idx = i * sz
            x_new = x_new - self.DX[:, start_idx:end_idx] @ (self.DF[:, start_idx:end_idx].T @ f1)

        # Deal with the last group - compute N
        if self.EN_stage != 1:
            # Initialize or resize N if needed
            n_size = x1.size
            if self.N is None or self.N.shape[1] < res:
                self.N = np.zeros((n_size, sz))
            
            self.N[:, res-1] = -mix * dx
            
            for i in range(1, ngroup + 1):
                start_idx = (i - 1) * sz
                end_idx = i * sz
                self.N[:, res-1] = self.N[:, res-1] + \
                    self.DF[:, start_idx:end_idx] @ (self.DX[:, start_idx:end_idx].T @ dx)

        # Apply the last group contribution
        if self.EN_stage == 1 and res == sz:
            start_idx = ngroup * sz
            end_idx = m
            if end_idx > start_idx:
                x_new = x_new - self.DX[:, start_idx:end_idx] @ (self.DF[:, start_idx:end_idx].T @ f1)
        else:
            M = self.N[:, :res].T @ self.DF[:, m-res:m]
            
            C = qinv(M, tol)
            
            if res == sz:
                # This updates DF to become V in the note
                self.DF[:, m-sz:m] = self.N[:, :sz] @ C.T
                
                x_new = x_new - self.DX[:, m-res:m] @ (self.DF[:, m-res:m].T @ f1)
            else:
                x_new = x_new - self.DX[:, m-res:m] @ ((C @ self.N[:, :res].T) @ f1)

        # Store current iterate for next iteration
        if self.EN_stage != 2:
            self.DX = np.hstack([self.DX, x1[:, np.newaxis]])
            self.DF = np.hstack([self.DF, f1[:, np.newaxis]])

        # Update EN_stage for next iteration
        if self.EN_stage == 1:
            self.EN_stage = 2
        elif self.EN_stage == 2:
            self.EN_stage = 1

        return x_new, m


# Module-level persistent mixer instance
_persistent_mixer = None


def get_mixer(reset=False):
    """Get or create the persistent mixer instance."""
    global _persistent_mixer
    if reset or _persistent_mixer is None:
        _persistent_mixer = mixer()
    return _persistent_mixer


def mixer_step(x1, f1, reset=False):
    """Convenience function."""
    return get_mixer(reset=reset).mixer(x1, f1)
