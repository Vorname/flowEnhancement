import numpy as np
from array import array

class FlowData:
    __file = None
    __fileName = None
    __frames = None
    __nb = None
    __index = None
    __clean = False

    def frames(self):
        return self.__frames

    def __offset(self, i):
        return 8 + i*8*self.__nb;

    def __init__(self, fileName, read=True, number_basis=0, frames=0):
        self.__fileName=fileName
        self.__index=0
        if read:
            self.__file = open(fileName, "br")
            nb_fc=array("i", self.__file.read(8))
            self.__nb=nb_fc[0]
            self.__frames=nb_fc[1]
        else:
            self.__file = open(fileName, "bw+")
            self.__nb = (number_basis+1)**2
            self.__frames = frames
            self.__file.write(array("i", [self.__nb, self.__frames]))

    def __del__(self):
        self.Close()

    def Close(self):
        if not self.__clean:
            self.__file.close()
            self.__clean=True

    def __getitem__(self, key):
        if self.__clean:
            raise Exception("file already closed")
        if key < 0 or key >= self.__frames:
            raise Exception("index out of range")
        if key != self.__index:
            self.__file.seek(self.__offset(key), 0)
            self.__index=key
        self.__index += 1
        return np.array(array("d", self.__file.read(8*self.__nb)));

    def AddFrameData(self, data ):
        if self.__clean:
            raise Exception("file already closed")
        if self.__index > self.__frames:
            raise Exception("added too many frames")
        if len(data) > self.__nb:
            raise Exception("more basisfunctions than expected")
        self.__file.write(data)
        self.__index += 1
