import numpy as np
from integration import *
from global_functions import *
import unit_test
from Log import *
import os
from flow_data import FlowData

grid = 256;
number_basis_functions = 5;
t_start=0;
t_end=10;
fps=20;

PrintInfo("Intitializing random flow and Advection Matrix")
time_index = StartClock();

Advection_Matrix = BuildAdvectionMatrix(number_basis_functions, printProgress=True, time_index=time_index);

PrintInfo("Intitializing of the advection matrix took {}s".format(GetTime(time_index)))
x = np.linspace(0, np.pi, grid);
y = np.linspace(0, np.pi, grid);
x1, x2 = np.meshgrid(x,y);

basisfunctions = GetBasisfunctions(number_basis_functions, x1, x2);

random_flow = np.zeros([number_basis_functions+1, number_basis_functions+1]);

random_flow[1:number_basis_functions+1, 1:number_basis_functions+1] = np.random.randn(number_basis_functions, number_basis_functions)

random_flow[5:,5:] = 0;

random_flow = random_flow.reshape((number_basis_functions+1)*(number_basis_functions+1))
for i in range(1, number_basis_functions+1):
    for j in range(1, number_basis_functions+1):
        random_flow[i*number_basis_functions+j] = 1/-lambda_k(i,j)*random_flow[i*number_basis_functions+j]

PrintInfo("Intitialization took {}s".format(GetTime(time_index)))
print()
fd = FlowData("../data/images/Matrix_Free/Random_flow_velocity.fd", False, number_basis_functions, fps*(t_end-t_start));
ReStartClock(time_index);
Integrate(random_flow, Advection_Matrix, basisfunctions, t_start, t_end, fps, x1, x2, "../data/images/Matrix_Free/frames/Random_flow_velocity", timeindex = time_index, render=True, flow_data=fd);
os.system("rm ../data/images/Matrix_Free/Random_flow_velocity.mp4")
os.system("ffmpeg -r {} -start_number_range {} -f image2 -i ../data/images/Matrix_Free/frames/Random_flow_velocity%03d.png -vcodec libx264 -pix_fmt yuv420p ../data/images/Matrix_Free/Random_flow_velocity.mp4 &> /dev/null".format(fps, fps*(t_end-t_start)));
PrintInfo("Integration of the advection took {}s using Advection Matrix".format(GetTime(time_index)));
print()
fd.Close()
fd2 = FlowData("../data/images/Matrix_Free/Random_flow_velocity_matrix_free.fd", False, number_basis_functions, fps*(t_end-t_start));
ReStartClock(time_index);
Integrate(random_flow, Advection_Matrix, basisfunctions, t_start, t_end, fps, x1, x2, "../data/images/Matrix_Free/frames/Random_flow_velocity_matrix_free", usefft=True, timeindex=time_index, render=True, flow_data=fd2);
os.system("rm ../data/images/Matrix_Free/Random_flow_velocity_matrix_free.mp4")
os.system("ffmpeg -r {} -start_number_range {} -f image2 -i ../data/images/Matrix_Free/frames/Random_flow_velocity_matrix_free%03d.png -vcodec libx264 -pix_fmt yuv420p ../data/images/Matrix_Free/Random_flow_velocity_matrix_free.mp4 &> /dev/null".format(fps, fps*(t_end-t_start)));
PrintInfo("Integration of the advection took {}s without Advection Matrix\n".format(GetTime(time_index)));
fd2.Close()
