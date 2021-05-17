import time
from global_functions import BuildAdvectionMatrix
import numpy as np
from Log import ProgressBar as pb

def main():
    print()
    nb = 15
    k = 1000

    AdvM = BuildAdvectionMatrix(nb)
    c = np.zeros([nb+1, nb+1])
    c[1:,1:] = np.random.randn(nb, nb)
    c = c.reshape((nb+1)**2)

    c1 = np.copy(c)

    start = time.perf_counter()
    for i in range(k):
        pb.ProgressBar(i, k, "")
        c1 += 1.0/k * AdvM.Apply(c1)
    print("Seriell took {:0.2f}s\n".format(time.perf_counter()-start))
    start = time.perf_counter()
    for i in range(k):
        pb.ProgressBar(i, k, "")
        c += 1.0/k * AdvM.Apply_Parallel(c)
    print("Parallel took {:0.2f}s\n".format(time.perf_counter() - start))

    print("Error of Parallel = {}".format(np.linalg.norm(c-c1)))


if __name__ == "__main__":
    main()
