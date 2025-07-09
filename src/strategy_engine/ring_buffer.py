import numpy as np
from numba import njit

# ────────────────────────────────────────────────────────────────────────
# Ultra‑fast RingBuffer  v1.0
#   • Public API **superset** of the original → nothing breaks
#   • push‑side kernel & NEW dump‑kernel are fully ``@njit``‑compiled
#   • NEW helper ``get_last_prices()`` returns the Last‑price window in
#     newest‑first order without any dtype promotion or object overhead.
#   • ``get_features_array()`` kept for backward compatibility but now
#     delegates to the dump kernels to avoid Python loops.
# ────────────────────────────────────────────────────────────────────────

# ---------- low‑level compiled helpers ---------------------------------
@njit(cache=True)
def _push_kernel(lp_arr, ss_arr, uid_arr, dt_arr, head, size,
                 last_price, sec_sm, uid, dt_int):
    lp_arr[head]  = last_price
    ss_arr[head]  = sec_sm
    uid_arr[head] = uid
    dt_arr[head]  = dt_int
    head += 1
    if head == size:
        head = 0
    return head

@njit(cache=True)
def _reverse_dump(lp_arr, ss_arr, uid_arr, dt_arr, head, count, size):
    """Return the data in newest‑first order.

    Produces four *views* (copies, but contiguous) for lp, ss, uid, dt.
    No Python objects inside Numba.
    """
    n = count if count < size else size
    out_lp  = np.empty(n, dtype=np.float32)
    out_ss  = np.empty(n, dtype=np.float32)
    out_uid = np.empty(n, dtype=np.int32)
    out_dt  = np.empty(n, dtype=np.int64)

    if count < size:
        for i in range(n):
            j = n - 1 - i           # reverse contiguous slice
            out_lp[i]  = lp_arr[j]
            out_ss[i]  = ss_arr[j]
            out_uid[i] = uid_arr[j]
            out_dt[i]  = dt_arr[j]
    else:
        idx = head - 1
        for i in range(n):
            if idx < 0:
                idx += size
            out_lp[i]  = lp_arr[idx]
            out_ss[i]  = ss_arr[idx]
            out_uid[i] = uid_arr[idx]
            out_dt[i]  = dt_arr[idx]
            idx -= 1
    return out_lp, out_ss, out_uid, out_dt

# ---------- public RingBuffer ------------------------------------------
class RingBuffer:
    __slots__ = ("size", "_lp", "_ss", "_uid", "_dt", "_vol", "_head", "_count")

    def __init__(self, *, max_size: int):
        self.size   = int(max_size)
        self._lp    = np.empty(self.size, dtype=np.float32)
        self._ss    = np.empty(self.size, dtype=np.float32)
        self._uid   = np.empty(self.size, dtype=np.int32)
        self._dt    = np.empty(self.size, dtype=np.int64)
        self._vol = np.zeros(self.size, dtype=np.float64)
        self._head  = 0
        self._count = 0

    # ────────────────────────────────────────────────────────────────
    def push(self, last_price: float, sec_sm: float, uid: int, dt, volume: float):
        dt_int = np.int64(np.datetime64(dt, "ns").view(np.int64))
        self._head = _push_kernel(self._lp, self._ss, self._uid, self._dt,
                                  self._head, self.size,
                                  last_price, sec_sm, uid,
                                  np.int64(np.datetime64(dt, "ns").view(np.int64)))
        if self._count < self.size:
            self._count += 1
        # store accumulated volume at the just‐written slot
        idx = (self._head - 1) % self.size
        self._vol[idx] = volume

    def update_last_volume(self, volume: float):
        """Overwrite the most recently pushed volume."""
        idx = (self._head - 1) % self.size
        self._vol[idx] = volume

    # ────────────────────────────────────────────────────────────────
    def is_ready(self):
        return self._count == self.size

    # ────────────────────────────────────────────────────────────────
    def reset(self):
        self._head  = 0
        self._count = 0

    # ────────────────────────────────────────────────────────────────
    # NEW ultra‑cheap accessors --------------------------------------
    # ────────────────────────────────────────────────────────────────
    def get_last_prices(self):
        """Float32 view of Last prices (newest‑first)."""
        lp, _, _, _ = _reverse_dump(self._lp, self._ss, self._uid, self._dt,
                                    self._head, self._count, self.size)
        return lp

    def get_last_window(self):
        """Returns (*lp*, *ss*, *uid*, *dt*) ndarray tuple, newest‑first."""
        lp_w, ss_w, uid_w, dt_w = _reverse_dump(
            self._lp, self._ss, self._uid, self._dt,
            self._head, self._count, self.size
        )
        # build volume window in newest-first order
        n = lp_w.shape[0]
        vol_w = np.empty(n, dtype=self._vol.dtype)
        for i in range(n):
            if self._count < self.size:
                idx = n - 1 - i
            else:
                idx = (self._head - 1 - i) % self.size
            vol_w[i] = self._vol[idx]
        return lp_w, ss_w, uid_w, dt_w, vol_w

    # ────────────────────────────────────────────────────────────────
    def get_features_array(self):
        """Backward‑compat object array for legacy code (slower)."""
        lp, ss, uid, dt, vol = self.get_last_window()
        out = np.empty((lp.shape[0], 4), dtype=object)
        out[:, 0] = lp
        out[:, 1] = ss
        out[:, 2] = uid
        out[:, 3] = dt.view("datetime64[ns]")
        out[:, 4] = vol  
        return out
