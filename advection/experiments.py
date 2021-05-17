from render import Render
from flow_data import FlowData
from Log import *
import unit_test
from integration import Integrate
from global_functions import BuildAdvectionMatrix
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as plt
import sys, getopt

def GetArguments(args):
    t_end = 20
    fps = 10
    start_number_bf = 5
    end_for_matrix = 15
    end_number_bf = 50
    step_width = 1

    try:
        opts, args = getopt.getopt(args, "ht:n:s:m:e:w:", ["t_end=","fps=","start_number_bf=","end_for_matrix=","end_number_bf=","step_width"])
    except getopt.GetoptError:
        print("experiments.py -t <end_time> -n <fps> -s <number of start basis> -m <last number of basis functions for matrices> -e <end number of basises> -w <step_width>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("experiments.py -t <end_time> -n <fps> -s <number of start basis> -m <last number of basis functions for matrices> -e <end number of basises> -w <step_width>")
            sys.exit()
        elif opt in ("-t", "--t_end"):
            t_end = int(arg)
        elif opt in ("-n", "--fps"):
            fps = int(arg)
        elif opt in ("-s", "--start_number_bf"):
            start_number_bf = int(arg)
        elif opt in ("-m", "--end_for_matrix"):
            end_for_matrix = int(arg)
        elif opt in ("-e", "--end_number_bf"):
            end_number_bf = int(arg)
        elif opt in ("-w", "--step_width"):
            step_width = int(arg)

    return [t_end, fps, start_number_bf, end_for_matrix, end_number_bf, step_width]

def main(args):
    [t_end, fps, start_number_bf, end_for_matrix, end_number_bf, step_width] = GetArguments(args)

    PrintInfo("experiments.py -t {} -n {} -s {} -m {} -e {} -w {}".format(t_end, fps, start_number_bf, end_for_matrix, end_number_bf, step_width))

    flow_base = np.zeros([start_number_bf+1, start_number_bf+1])
    flow_base[1:,1:] = np.random.randn(start_number_bf, start_number_bf)
    clock_handle_overall = StartClock()
    clock_handle_current = StartClock()
    plot_data = np.zeros([6, int((end_number_bf-start_number_bf)/step_width)])

    energy_plot = np.zeros([5, 1+int(t_end * fps)])

    energy_plot[0] = [float(i)/fps for i in range(1+int(t_end*fps))]

    i= 0;
    j = 0;
    get_energy = int((end_for_matrix - start_number_bf)/step_width)
    for nbf in range(start_number_bf, end_number_bf, step_width):
        print()
        PrintInfo("Begin simulation with {} basisfunctions\n".format(nbf))
        flow = np.zeros([nbf+1,nbf+1])
        flow[0:start_number_bf+1, 0:start_number_bf+1]=flow_base
        flow = flow.reshape((nbf+1)**2)
        plot_data[0, i] = nbf
        fd2 = FlowData("../data/experiments/FFT{:03}.fd".format(nbf), read=False, number_basis = nbf, frames =1+ t_end*fps)

        if nbf <= end_for_matrix:
            fd1 = FlowData("../data/experiments/Matrix{:03}.fd".format(nbf), read=False, number_basis = nbf, frames =1+ t_end*fps)

            ReStartClock(clock_handle_current)
            AdvM = BuildAdvectionMatrix(nbf, printProgress=True, time_index=clock_handle_current)
            plot_data[1, i] = GetTime(clock_handle_current)
            PrintInfo("Building Advection Matrix took {:10.2f}s".format(plot_data[1,i]))
            plot_data[4,i] = (AdvM.Nonzeroentries()/(nbf+1)**6)*100.0
            plot_data[5,i] = AdvM.Nonzeroentries()
            PrintInfo("Start Advection with Matrix")
            print()
            ReStartClock(clock_handle_current)
            Integrate(flow, AdvM, None, 0, t_end, fps, None, None, None, render=False, timeindex=clock_handle_current, flow_data=fd1, usefft=False, energy_preserving=True)
            plot_data[2,i] = GetTime(clock_handle_current)
            PrintInfo("Integration over the Advection using the Matrix took {:10.2f}s".format(plot_data[2,i]))
            fd1.Close()

            if nbf == get_energy:
                PrintInfo("Begin Energy collection")
                leap_frog_Matrix = FlowData("../data/experiments/LFM{:03}.fd".format(nbf), read=False, number_basis=nbf, frames=1+t_end*fps)
                runge_kutta_Matrix = FlowData("../data/experiments/RKM{:03}.fd".format(nbf), read=False, number_basis=nbf, frames=1+t_end*fps)
                leap_frog_fft = FlowData("../data/experiments/LFF{:03}.fd".format(nbf), read=False, number_basis=nbf, frames=1+t_end*fps)
                runge_kutta_fft = FlowData("../data/experiments/RKF{:03}.fd".format(nbf), read=False, number_basis=nbf, frames=1+t_end*fps)
                print()
                ReStartClock(clock_handle_current)
                Integrate(flow, AdvM, None, 0, t_end, fps, None, None, None, render=False, usefft=False, timeindex=clock_handle_current, energy_preserving=False, energy_data=energy_plot[1], flow_data=leap_frog_Matrix)
                print()
                ReStartClock(clock_handle_current)
                Integrate(flow, AdvM, None, 0, t_end, fps, None, None, None, render=False, usefft=False, timeindex=clock_handle_current, energy_preserving=True, energy_data=energy_plot[2], flow_data=runge_kutta_Matrix)
                print()
                ReStartClock(clock_handle_current)
                Integrate(flow, AdvM, None, 0, t_end, fps, None, None, None, render=False, usefft=True, timeindex=clock_handle_current, energy_preserving=False, energy_data=energy_plot[3], flow_data=leap_frog_fft)
                print()
                ReStartClock(clock_handle_current)
                Integrate(flow, AdvM, None, 0, t_end, fps, None, None, None, render=False, usefft=True, timeindex=clock_handle_current, energy_preserving=True, energy_data=energy_plot[4], flow_data=runge_kutta_fft)
                print()
                leap_frog_Matrix.Close()
                runge_kutta_Matrix.Close()
                leap_frog_fft.Close()
                runge_kutta_fft.Close()

            j += 1

        PrintInfo("Start Advection without Matrix\n")
        ReStartClock(clock_handle_current)
        Integrate(flow, AdvM, None, 0, t_end, fps, None, None, None, render=False, timeindex=clock_handle_current, flow_data=fd2, usefft=True, energy_preserving=True)
        plot_data[3,i] = GetTime(clock_handle_current)
        PrintInfo("Integration over Advection without using the Matrix took {:10.2f}s".format(plot_data[3,i]))
        fd2.Close()

        i += 1;
    PrintInfo("Running all experiments took {:10.2f}s".format(GetTime(clock_handle_overall)))

    time_fig = go.Figure(layout=dict(title=dict(text="Times for the Advection calculation in seconds"), xaxis=dict(title="Number of basis functions per dimension"),yaxis=dict(title="seconds")))
    time_fig.add_trace(go.Scatter(x=plot_data[0,0:j], y=plot_data[1,0:j], mode='lines', name='Building Time of Advection Matrix'))
    time_fig.add_trace(go.Scatter(x=plot_data[0,0:j], y=plot_data[2,0:j], mode='lines', name='Advection using Matrix'))
    time_fig.add_trace(go.Scatter(x=plot_data[0], y=plot_data[3], mode='lines', name='Advection using FFT'))
    time_fig.write_image("../data/experiments/times.pdf")
    plt.plot(time_fig, filename="../data/experiments/times.html")

    energy_fig = go.Figure(layout=dict(title=dict(text="Energy Evolution"), xaxis=dict(title="seconds"),yaxis=dict(title="")))
    energy_fig.add_trace(go.Scatter(x=energy_plot[0], y=energy_plot[1], mode='lines', name='Leap frog Matrix'))
    energy_fig.add_trace(go.Scatter(x=energy_plot[0], y=energy_plot[2], mode='lines', name='Runge Kutta Matrix'))
    energy_fig.add_trace(go.Scatter(x=energy_plot[0], y=energy_plot[3], mode='lines', name='Leap frog FFT'))
    energy_fig.add_trace(go.Scatter(x=energy_plot[0], y=energy_plot[4], mode='lines', name='Runge Kutta FFT'))
    energy_fig.write_image("../data/experiments/energy.pdf")
    plt.plot(energy_fig, filename="../data/experiments/energy.html")

    non_zero_fig = make_subplots(rows=2, cols = 1)
    non_zero_fig.append_trace(go.Scatter(x=plot_data[0,0:j], y=plot_data[5,0:j], mode="lines", name="Number of non zero entries"), row=1,col=1)
    non_zero_fig.append_trace(go.Scatter(x=plot_data[0,0:j], y=plot_data[4,0:j], mode="lines", name="Percent of non zero entries"), row=2,col=1)

    non_zero_fig.update_layout(title=dict(text="Number of non zero entries of the advection Matrix"))
    non_zero_fig.update_xaxes(title_text="Number basis functions per dimension", row=1, col=1)
    non_zero_fig.update_xaxes(title_text="Number basis functions per dimension", row=2, col=1)

    non_zero_fig.update_yaxes(title_text="Number non zero entries", row=1, col=1)
    non_zero_fig.update_yaxes(title_text="Percent of non zero entries", row=2, col=1)

    non_zero_fig.write_image("../data/experiments/nonzero.pdf")
    plt.plot(non_zero_fig, filename="../data/experiments/nonzero.html")

if __name__ == "__main__":
    main(sys.argv[1:])
