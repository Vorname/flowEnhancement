import numpy as np
import time

def MatrixToString(mat):
    res = "[";
    #max_length = 0;
    #for i in range(mat.shape[0]):
    #    for j in range(mat.shape[1]):
    #        if(len(str(mat[i,j])) > max_length):
    #            max_length = len(str(mat[i,j]));
    for i in range(mat.shape[0]):
        res += "["
        for j in range(mat.shape[1]):
            #tmp = str(mat[i,j]);
            tmp = '{:06.4f}'.format(mat[i,j])
            #for k in range(len(tmp), max_length):
            #    tmp += " ";
            res += tmp + " ";
        res += "]\n";

    return res + "]\n\n";

def __format(message, newLineStartings):
    arr = message.splitlines()
    m = arr[0];
    for i in range(1, len(arr)):
        m += "\n"+newLineStartings+arr[i]
    return m

def PrintInfo(message):
    m = __format(message, "      ")
    print("\033[36mInfo:\033[0m " + m);

class LoggerClass:
    def __init__(self, fileName="file.log"):
        self.file = open(fileName,"w+");
        pass

    def __del__(self):
        self.file.close();

    def Log(self, message):
        self.file.write(str(message) + "\n")

    def LogMatrix(self, mat ):
        self.file.write(MatrixToString(mat));

__Clock_Array=[];

class ProgressBar:
    def PrintTestResult(bool_value, test_name, log = None):
        if bool_value:
            print("\033[32mPassed\033[0m  {}".format(test_name));
        elif log is not None:
            print("\033[31mFailed\033[0m  {} -- {}".format(test_name, log));
        else:
            print("\033[31mFailed\033[0m  {}".format(test_name));

    def ProgressBar(current_step, max_steps, Message, pressicion = 2, time_index=-1):
        if time_index >= 0:
            form = "\033[1A\033[K{:10."+str(pressicion)+"%} completed within {:10}s {}";
            print(form.format(current_step*1.0/max_steps, int(GetTime(time_index)*100)/100.0, Message))
        else:
            form = "\033[1A\033[K{:10."+str(pressicion)+"%} completed {}";
            print(form.format(current_step*1.0/max_steps, Message))

def StartClock():
    index = len(__Clock_Array)
    __Clock_Array.append(time.perf_counter())
    return index;
def ReStartClock(index):
    __Clock_Array[index] = time.perf_counter()
def GetTime(index):
    return time.perf_counter()-__Clock_Array[index]


Logger = LoggerClass()
