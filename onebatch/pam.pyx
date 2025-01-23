import cython
from cpython.array cimport array, clone


@cython.boundscheck(False)
@cython.wraparound(False)
def swap_eager(float[:, ::1] Dist, int[::1] medoids_init, int K, int n_swap, int N, int B, float tol_init):
    
    cdef array[float] templatef = array('f')
    cdef array[int] templatei = array('i')

    cdef float swap_gain = 0
    cdef float swap_gain_0 = 0
    cdef float tol
    
    cdef int index, index1, index2
    cdef int i, j, k, u, v, s
    cdef float first_dist_init = 10000000
    cdef float sec_dist_init = 10000000
    cdef float first_dist
    cdef float sec_dist
    cdef float loss = 0
    cdef int last_change = -1
    cdef bint not_medoid_init = 1
    cdef bint not_medoid
    cdef bint finish = 0

    cdef float[::1] min_Dist_to_med = clone(templatef, B, False)
    cdef float[::1] second_min_Dist_to_med = clone(templatef, B, False)
    cdef float[::1] swap_gains_K = clone(templatef, K, True)
    cdef float[::1] swap_gains_K_copy = clone(templatef, K, False)
    cdef int[::1] nearest = clone(templatei, B, False)
    cdef int[::1] second = clone(templatei, B, False)
    cdef int[::1] medoids = clone(templatei, K, False)

    with nogil:
        medoids[:] = medoids_init
    
        for j in range(B):
            index1 = 0
            index2 = 0
            first_dist = first_dist_init
            sec_dist = sec_dist_init
            for k in range(K):
                if Dist[medoids[k], j] < sec_dist:
                    if Dist[medoids[k], j] <= first_dist:
                        index2 = index1
                        index1 = k
                        sec_dist = first_dist
                        first_dist = Dist[medoids[k], j]
                    else:
                        index2 = k
                        sec_dist = Dist[medoids[k], j]
            min_Dist_to_med[j] = first_dist
            second_min_Dist_to_med[j] = sec_dist
            nearest[j] = index1
            second[j] = index2
        
        for j in range(B):
            k = nearest[j]
            swap_gains_K[k] += min_Dist_to_med[j] - second_min_Dist_to_med[j]
            loss += min_Dist_to_med[j]
        
        tol = tol_init * loss
        
        for s in range(n_swap):
            
            for i in range(N):
                if i == last_change:
                    finish = 1
                    break
    
                not_medoid = not_medoid_init
                for k in range(K):
                    if i == medoids[k]:
                        not_medoid = 0
                
                if not_medoid:
                    swap_gain = swap_gain_0
                    swap_gains_K_copy[:] = swap_gains_K
                    for j in range(B):
                        k = nearest[j]
                        if Dist[i, j] < min_Dist_to_med[j]:
                            swap_gains_K_copy[k] += second_min_Dist_to_med[j] - min_Dist_to_med[j]
                            swap_gain += min_Dist_to_med[j] - Dist[i, j]
                        elif Dist[i, j] < second_min_Dist_to_med[j]:
                            swap_gains_K_copy[k] += second_min_Dist_to_med[j] - Dist[i, j]
            
                    index = 0
                    for k in range(K):
                        if swap_gains_K_copy[k] > swap_gains_K_copy[index]:
                            index = k
            
                    swap_gain += swap_gains_K_copy[index]
    
                    if swap_gain > tol:
                        medoids[index] = i
                        last_change = i
    
                        swap_gains_K[index] = swap_gain_0
                        loss -= swap_gain
                        tol = tol_init * loss
    
                        for j in range(B):
                            if ((nearest[j] == index) or
                                (second[j] == index) or
                                (Dist[i, j] <= second_min_Dist_to_med[j])):
                            
                                k = nearest[j]
                                if k != index:
                                    swap_gains_K[k] += second_min_Dist_to_med[j] - min_Dist_to_med[j]
        
                                index1 = 0
                                index2 = 0
                                first_dist = first_dist_init
                                sec_dist = sec_dist_init
                                for k in range(K):
                                    if Dist[medoids[k], j] < sec_dist:
                                        if Dist[medoids[k], j] <= first_dist:
                                            index2 = index1
                                            index1 = k
                                            sec_dist = first_dist
                                            first_dist = Dist[medoids[k], j]
                                        else:
                                            index2 = k
                                            sec_dist = Dist[medoids[k], j]
                                min_Dist_to_med[j] = first_dist
                                second_min_Dist_to_med[j] = sec_dist
                                nearest[j] = index1
                                second[j] = index2
                            
                                k = nearest[j]
                                swap_gains_K[k] += min_Dist_to_med[j] - second_min_Dist_to_med[j]
    
            if finish or last_change == -1:
                break
    
    result_as_list = [m for m in medoids]
    return result_as_list