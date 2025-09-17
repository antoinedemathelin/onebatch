# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
import cython
from cpython.array cimport array, clone
import numpy as np
from cython.parallel cimport prange, threadid
cimport openmp


cdef inline void _init_for_sample_j(
    int j,
    int K,
    float[:, ::1] Dist,
    int[::1] medoids,
    float first_dist_init,
    float sec_dist_init,
    float[::1] min_Dist_to_med,
    float[::1] second_min_Dist_to_med,
    int[::1] nearest,
    int[::1] second,
    float[:, ::1] swap_gains_tls_mv,
    float[::1] loss_tls_mv
) nogil noexcept:
    cdef int kk, idx1, idx2, tid
    cdef float first_dist, sec_dist
    idx1 = 0
    idx2 = 0
    first_dist = first_dist_init
    sec_dist = sec_dist_init
    for kk in range(K):
        if Dist[medoids[kk], j] < sec_dist:
            if Dist[medoids[kk], j] <= first_dist:
                idx2 = idx1
                idx1 = kk
                sec_dist = first_dist
                first_dist = Dist[medoids[kk], j]
            else:
                idx2 = kk
                sec_dist = Dist[medoids[kk], j]
    min_Dist_to_med[j] = first_dist
    second_min_Dist_to_med[j] = sec_dist
    nearest[j] = idx1
    second[j] = idx2

    tid = threadid()
    swap_gains_tls_mv[tid, idx1] += first_dist - sec_dist
    loss_tls_mv[tid] += first_dist


cdef inline void _evaluate_candidate_i(
    int i,
    int K,
    int B,
    char[::1] is_medoid,
    float[:, ::1] Dist,
    float[::1] min_Dist_to_med,
    float[::1] second_min_Dist_to_med,
    int[::1] nearest,
    float[:, ::1] delta_k_tls_mv,
    float[::1] swap_gains_K,
    float[::1] best_gain_tls_mv,
    int[::1] best_i_tls_mv,
    int[::1] best_k_tls_mv
) nogil noexcept:
    cdef int kk, kk2, j, kn, k_best, tid
    cdef float d, mn, sc, swap_gain_add_i, over_k, v, gain_i

    if is_medoid[i] != 0:
        return

    tid = threadid()
    swap_gain_add_i = 0.0

    for kk in range(K):
        delta_k_tls_mv[tid, kk] = 0.0

    for j in range(B):
        d = Dist[i, j]
        mn = min_Dist_to_med[j]
        sc = second_min_Dist_to_med[j]
        kn = nearest[j]
        if d < mn:
            swap_gain_add_i += (mn - d)
            delta_k_tls_mv[tid, kn] += (sc - mn)
        elif d < sc:
            delta_k_tls_mv[tid, kn] += (sc - d)

    k_best = 0
    over_k = swap_gains_K[0] + delta_k_tls_mv[tid, 0]
    for kk2 in range(1, K):
        v = swap_gains_K[kk2] + delta_k_tls_mv[tid, kk2]
        if v > over_k or (v == over_k and kk2 < k_best):
            over_k = v
            k_best = kk2

    gain_i = swap_gain_add_i + over_k

    if (gain_i > best_gain_tls_mv[tid] or
        (gain_i == best_gain_tls_mv[tid] and (
            best_i_tls_mv[tid] == -1 or i < best_i_tls_mv[tid] or
            (i == best_i_tls_mv[tid] and k_best < best_k_tls_mv[tid])
        ))):
        best_gain_tls_mv[tid] = gain_i
        best_i_tls_mv[tid] = i
        best_k_tls_mv[tid] = k_best


cdef inline void _recompute_after_swap_for_j(
    int j,
    int K,
    int best_k,
    int best_i,
    float[:, ::1] Dist,
    int[::1] medoids,
    float first_dist_init,
    float sec_dist_init,
    float[::1] min_Dist_to_med,
    float[::1] second_min_Dist_to_med,
    int[::1] nearest,
    int[::1] second,
    float[:, ::1] swap_gains_tls_mv
) nogil noexcept:
    cdef int kk, idx1, idx2, tid, k_old
    cdef float first_dist, sec_dist

    if (nearest[j] == best_k) or (second[j] == best_k) or (Dist[best_i, j] <= second_min_Dist_to_med[j]):
        tid = threadid()
        k_old = nearest[j]
        if k_old != best_k:
            swap_gains_tls_mv[tid, k_old] += (second_min_Dist_to_med[j] - min_Dist_to_med[j])

        idx1 = 0
        idx2 = 0
        first_dist = first_dist_init
        sec_dist = sec_dist_init
        for kk in range(K):
            if Dist[medoids[kk], j] < sec_dist:
                if Dist[medoids[kk], j] <= first_dist:
                    idx2 = idx1
                    idx1 = kk
                    sec_dist = first_dist
                    first_dist = Dist[medoids[kk], j]
                else:
                    idx2 = kk
                    sec_dist = Dist[medoids[kk], j]
        min_Dist_to_med[j] = first_dist
        second_min_Dist_to_med[j] = sec_dist
        nearest[j] = idx1
        second[j] = idx2

        swap_gains_tls_mv[tid, idx1] += (first_dist - sec_dist)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def swap_eager(float[:, ::1] Dist, int[::1] medoids_init, int K, int max_iter, int N, int B, float tol_init, int n_threads=0):
    """
    Best-improvement PAM with OpenMP parallelization.
    Keeps outputs deterministic via fixed tie-breaking (prefer smaller i, then smaller k on equal gain).
    """
    cdef array[float] templatef = array('f')
    cdef array[int] templatei = array('i')
    cdef array[char] templateb = array('b')

    cdef float tol
    cdef float first_dist_init = 10000000
    cdef float sec_dist_init = 10000000

    cdef int i, j, s
    cdef int T

    # Working arrays (1D)
    cdef float[::1] min_Dist_to_med = clone(templatef, B, False)
    cdef float[::1] second_min_Dist_to_med = clone(templatef, B, False)
    cdef float[::1] swap_gains_K = clone(templatef, K, True)
    cdef int[::1] nearest = clone(templatei, B, False)
    cdef int[::1] second = clone(templatei, B, False)
    cdef int[::1] medoids = clone(templatei, K, False)
    cdef char[::1] is_medoid = clone(templateb, N, False)

    # Determine thread count
    if n_threads is None or n_threads <= 0:
        T = openmp.omp_get_max_threads()
    else:
        T = n_threads
    if T < 1:
        T = 1

    # TLS buffers allocated with NumPy (GIL region)
    delta_k_tls = np.zeros((T, K), dtype=np.float32)
    swap_gains_tls = np.zeros((T, K), dtype=np.float32)
    loss_tls = np.zeros((T,), dtype=np.float32)
    best_gain_tls = np.empty((T,), dtype=np.float32)
    best_i_tls = np.empty((T,), dtype=np.int32)
    best_k_tls = np.empty((T,), dtype=np.int32)

    cdef float[:, ::1] delta_k_tls_mv = delta_k_tls
    cdef float[:, ::1] swap_gains_tls_mv = swap_gains_tls
    cdef float[::1] loss_tls_mv = loss_tls
    cdef float[::1] best_gain_tls_mv = best_gain_tls
    cdef int[::1] best_i_tls_mv = best_i_tls
    cdef int[::1] best_k_tls_mv = best_k_tls

    # Hoisted loop-local declarations to function scope
    cdef float best_gain
    cdef int best_i, best_k
    cdef int u, tt

    cdef float loss = 0.0

    with nogil:
        # Initialize medoids and flags
        medoids[:] = medoids_init
        for i in range(N):
            is_medoid[i] = 0
        for i in range(K):
            is_medoid[medoids[i]] = 1

        # Zero TLS accumulators
        for i in range(T):
            loss_tls_mv[i] = 0.0
            for j in range(K):
                swap_gains_tls_mv[i, j] = 0.0

        # Step 1: initialize nearest/second per sample j (parallel over j)
        for j in prange(B, schedule='static', num_threads=T):
            _init_for_sample_j(
                j, K, Dist, medoids,
                first_dist_init, sec_dist_init,
                min_Dist_to_med, second_min_Dist_to_med,
                nearest, second,
                swap_gains_tls_mv, loss_tls_mv
            )

        # Reduce TLS to globals
        loss = 0.0
        for i in range(T):
            loss += loss_tls_mv[i]
            for j in range(K):
                swap_gains_K[j] += swap_gains_tls_mv[i, j]
        tol = tol_init * loss

        # Main loop: best-improvement each round
        for s in range(max_iter):
            # using hoisted cdefs: best_gain, best_i, best_k, u, tt

            # Reset per-thread bests
            for i in range(T):
                best_gain_tls_mv[i] = -1e30
                best_i_tls_mv[i] = -1
                best_k_tls_mv[i] = -1

            # Phase C: evaluate all candidate i in parallel
            for i in prange(N, schedule='static', num_threads=T):
                _evaluate_candidate_i(
                    i, K, B, is_medoid, Dist,
                    min_Dist_to_med, second_min_Dist_to_med, nearest,
                    delta_k_tls_mv, swap_gains_K,
                    best_gain_tls_mv, best_i_tls_mv, best_k_tls_mv
                )

            # Serial reduction: pick global best (i*, k*)
            best_gain = -1e30
            best_i = -1
            best_k = -1
            for tt in range(T):
                if (best_gain_tls_mv[tt] > best_gain or
                    (best_gain_tls_mv[tt] == best_gain and (
                        best_i == -1 or best_i_tls_mv[tt] < best_i or
                        (best_i_tls_mv[tt] == best_i and best_k_tls_mv[tt] < best_k)
                    ))):
                    best_gain = best_gain_tls_mv[tt]
                    best_i = best_i_tls_mv[tt]
                    best_k = best_k_tls_mv[tt]

            if best_gain <= tol or best_i == -1 or best_k == -1:
                break

            # Apply swap (serial)
            u = medoids[best_k]
            medoids[best_k] = best_i
            is_medoid[u] = 0
            is_medoid[best_i] = 1
            loss -= best_gain
            tol = tol_init * loss
            # the replaced medoid's swap gain is reset
            swap_gains_K[best_k] = 0.0

            # Phase B/A update: adjust affected samples and accumulate deltas into TLS
            for i in range(T):
                for j in range(K):
                    swap_gains_tls_mv[i, j] = 0.0

            for j in prange(B, schedule='static', num_threads=T):
                _recompute_after_swap_for_j(
                    j, K, best_k, best_i, Dist, medoids,
                    first_dist_init, sec_dist_init,
                    min_Dist_to_med, second_min_Dist_to_med,
                    nearest, second, swap_gains_tls_mv
                )

            # reduce TLS deltas into global swap_gains_K
            for i in range(T):
                for j in range(K):
                    swap_gains_K[j] += swap_gains_tls_mv[i, j]

    # Build Python result
    result_as_list = [m for m in medoids]
    nearest_as_list = [n for n in nearest]
    dist_to_nearest = [d for d in min_Dist_to_med]
    sol = {
        "medoids": result_as_list,
        "nearest": nearest_as_list,
        "dist_to_nearest": dist_to_nearest,
        "loss": loss,
        "steps": s
    }
    return sol