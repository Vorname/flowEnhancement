from render import Render
from flow_data import FlowData
from Log import *
import unit_test
from integration import Integrate
from global_functions import BuildAdvectionMatrix
import numpy as np
import sys, getopt, os
import matplotlib.pyplot as plt

def GetArguments(args):
    t_start = 0
    t_end = 10
    fps = 20
    number_basis = 10
    try:
        opts, args = getopt.getopt(args, "he:f:n:", ["t_end=","fps=","number_bf="])
    except Exception as e:
        print("plot_energy.py \noptions: -e/--t_end     : end time in seconds\n          -f/--fps       : frames per second as an integer\n          -n/--number_bf : number of basisfunctions\n           -h             : show this message")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("plot_energy.py \noptions: -e/--t_end     : end time in seconds\n          -f/--fps       : frames per second as an integer\n          -n/--number_bf : number of basisfunctions\n           -h             : show this message")
            sys.exit()
        elif opt in ("-e", "--t_end"):
            t_end = int(arg)
        elif opt in ("-f", "--fps"):
            fps = int(arg)
        elif opt in ("-n", "--number_bf"):
            number_basis = int(arg)

    return [t_start, t_end, fps, number_basis]

def Main(args):
    [t_start, t_end, fps, number_bf] = GetArguments(args)

    energy_plot = np.zeros([3, 1+int(t_end * fps)])
    energy_plot[0] = [float(i)/fps for i in range(1+int(t_end*fps))]
    coeffs = np.zeros([number_bf+1, number_bf+1])
    coeffs[1:,1:] = np.random.randn(number_bf, number_bf)
    coeffs = coeffs.reshape((number_bf+1)**2)
    os.system("mkdir -p ../data/energy_plot/")
    flow_data_Leap = FlowData("../data/energy_plot/LeapFrog.fd", read=False, number_basis = number_bf, frames=1+ t_end*fps)
    flow_data_RK4 = FlowData("../data/energy_plot/RK4.fd", read=False, number_basis= number_bf, frames=1+ t_end*fps)

    clock_index = StartClock()
    Integrate(coeffs, None, None, 0, t_end, fps, None, None, None, usefft=True, timeindex=clock_index, render=False, flow_data=flow_data_Leap, energy_preserving=False, energy_data=energy_plot[1])
    ReStartClock(clock_index)
    print()
    Integrate(coeffs, None, None, 0, t_end, fps, None, None, None, usefft=True, timeindex=clock_index, render=False, flow_data=flow_data_RK4, energy_preserving=True, energy_data=energy_plot[2])

    plt.plot(energy_plot[0], energy_plot[1], label="Leap Frog")
    plt.plot(energy_plot[0], energy_plot[2], label="Runge Kutta")
    plt.xlabel("time in seconds")
    plt.ylabel("energy")
    plt.title("Energy plot")
    plt.legend()
    plt.savefig("../data/energy_plot/energy_plot.pdf")

if __name__ == "__main__":
    Main(sys.argv[1:])
