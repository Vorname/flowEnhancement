import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

def apply_parallel(mat_list , coeffs , processes=mp.cpu_count()):
    res = mp.Array("d", np.zeros(coeffs.shape[0]))
    count = min(processes, len(mat_list))
    p = [mp.Process(target=f, args = (mat_list, int(i*coeffs.shape[0]/count), int((i+1)*coeffs.shape[0]/count), coeffs, res)) for i in range(count)]

    for pp in p:
        pp.start()
    for pp in p:
        pp.join()

    return np.array(res[:])

def f(arr, s, e, r, res):
    for i in range(s, e):
        res[i] = (r.dot(arr[i].dot(r)))

class AdvectionMatrix:
    dim=3;
    shape=[1,1,1];
    _sparse_list=[];

    def __init__(self, shape ):
        if len(shape) != 3:
            raise Exception("Shape is ill formated. Need 3d Shape to initialize Advection Matrix");

        self.shape=shape;
        self._sparse_list=np.array([sp.dok_matrix((shape[1],shape[2])) for i in range(shape[0])]);

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._sparse_list[item]
        elif len(item)<=3:
            return self._sparse_list[item[0]][item[1:3]]
        elif len(item)==6:
            sqr = (int)(np.floor(np.sqrt(self.shape[0])));
            return self._sparse_list[item[0]*sqr+item[1]][item[2]*sqr+item[3],item[4]*sqr+item[5]]
        else:
            raise Exception("Index Error");

    def __str__(self):
        res = ""
        for i in range(len(self._sparse_list)):
            nz = self._sparse_list[i].nonzero()
            nz = np.array([nz[0], nz[1]])
            for j in range(nz.shape[1]):
                res += str((i,nz[0,j],nz[1,j])) + " " + str(self[i, nz[0,j], nz[1,j]]) + "\n"
        return res

    def __setitem__(self, key, item):
        if isinstance(key, int):
            self_sparse_list[key] = item;
        if len(key)<=3:
            self._sparse_list[key[0]][key[1:3]]=item;
        elif len(key)==6:
            sqr = (int)(np.floor(np.sqrt(self.shape[0])));
            self._sparse_list[key[0]*sqr+key[1]][key[2]*sqr+key[3],key[4]*sqr+key[5]]=item
        else:
            raise Exception("Index Error");

    def __sub__(self, o):
        if hasattr(o, '_sparse_list'):
            res = AdvectionMatrix(self.shape)
            res._sparse_list=self._sparse_list-o._sparse_list
            return res;
        else:
            res = np.zeros(self.shape);
            for i in range(self.shape[0]):
                res[i] = self[i,:,:]-o[i];
            return res

    def Nonzeroentries(self):
        return sum(list(map(lambda x: len(x.nonzero()[0]), self._sparse_list)))

    def Set(self, i1i2j1j2 , values ):
        tmp = values.reshape(values.shape[0]*values.shape[1]);
        i = i1i2j1j2[0]*values.shape[0]+i1i2j1j2[1];
        j = i1i2j1j2[2]*values.shape[0]+i1i2j1j2[3];
        for k in range(len(tmp)):
            if abs(tmp[k]) > 0.00000001:
                self._sparse_list[k][i,j]=tmp[k];

    # def Apply(self, coeffs ):
    #     res = np.zeros(coeffs.shape);
    #     for i in range(len(res)):
    #         res[i] = coeffs.dot(self._sparse_list[i].dot(coeffs));
    #     return res;

    def Apply(self, coeffs ):
        return apply_parallel(self._sparse_list, coeffs)


    def Norm(self):
        tmp = np.zeros(self.shape);
        for i in range(self.shape[0]):
            tmp[i] = self._sparse_list[i].toarray();
        return np.linalg.norm(tmp);
