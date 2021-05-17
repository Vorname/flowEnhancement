# global imports
import sys
import os
import numpy as np
import sys, getopt
# local imports
sys.path.append('../')
from advection.flow_data import FlowData
from advection.visualize_flow import *
from advection.reconstruction import *
from advection.Log import *




help_text = "render.py --flow_data <flow data file> --res_x [default 512] --res_y [default 512] --dir [default ./] --fps [default 20] --output [default flow.mp4]"

def GetArguments(args):
    flow_data_name = "data.fd"
    resolution = [512, 512]
    result_directory = "./"
    fps = 20
    video_name = "flow.mp4"

    try:
        opts, args = getopt.getopt(args, "h", ["flow_data=","res_x=","res_y=","dir=","fps=","output="])
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(help_text)
            sys.exit()
        elif opt in ("--flow_data"):
            flow_data_name = arg
        elif opt in ("--res_x"):
            resolution[0] = int(arg)
        elif opt in ("--res_y"):
            resolution[1] = int(arg)
        elif opt in ("--dir"):
            result_directory = arg
        elif opt in ("--fps"):
            fps = int(arg)
        elif opt in ("--output"):
            video_name = arg

    return [flow_data_name, resolution, result_directory, fps, video_name]

def RenderDirect(flowData, fps, video_name="flow.mp4", result_directory = "./", resolution = [512, 512], inflated = False):

    if os.system("ls {} | grep tmp".format(result_directory)) != 256:
        os.system("rm -r {}".format(result_directory + "/tmp"))
    framedata = result_directory + "/tmp"
    os.system("mkdir {}".format(framedata))
    framedata += "/frame"


    if not inflated:
        num_frames = np.shape(flowData)[0]
        num_basis = np.shape(flowData)[1]
    else:
        num_frames = np.shape(flowData)[1]
        num_basis = np.shape(flowData)[0]
        
    x = np.linspace(0, 1, resolution[0])
    y = np.linspace(0, 1, resolution[1])
    x1, x2 = np.meshgrid(x,y)

    l = int(np.log10(num_frames))
    print()

    ending = "{:0" +str(l+1) +"}.png"
    clock_index = StartClock()
    for i in range(num_frames):
        ProgressBar.ProgressBar(i, num_frames, "", time_index=clock_index)
        if not inflated:
            c = np.zeros((num_basis+1,num_basis+1))
            c[1:,1:]=np.copy(flowData[i,:,:])
        else:
            c = flowData[:,i]
        flow = Reconstruct_Velocity(c.flatten(), None, True, res=resolution)
        visualize_velocity(flow, x1, x2, framedata+ending.format(i))
    ProgressBar.ProgressBar(num_frames, num_frames, "", time_index=clock_index)
    video = result_directory + video_name
    frames = framedata + "%0"+str(l+1)+"d.png"
    to_execute="yes | ffmpeg -r {} -start_number_range {} -f image2 -i {} -vcodec libx264 -pix_fmt yuv420p {}".format(fps, num_frames, frames, video)
    PrintInfo(to_execute)
    os.system(to_execute)


def Render(args):
    flow_data_name, resolution, result_directory, fps, video_name = GetArguments(args)
    out_fps = 32 # set output fps to fixed 32
    step_width = int(fps/out_fps)
    print("loading {}".format(flow_data_name))
    fd=np.load(flow_data_name)

    if os.system("ls {} | grep tmp".format(result_directory)) != 256:
        os.system("rm -r tmp")
    framedata = result_directory + "/tmp"
    os.system("mkdir {}".format(framedata))
    framedata += "/frame"

    x = np.linspace(0, 1, resolution[0]);
    y = np.linspace(0, 1, resolution[1]);
    x1, x2 = np.meshgrid(x,y);

    frames = np.shape(fd)[1]
    l = int(np.log10(frames))
    print()

    ending = "{:0" +str(l+1) +"}.png"
    clock_index = StartClock()
    count_i = 0
    # use every nth frame to get only 32 frames per second. speeds things up.
    for i in range(0,frames,step_width):
        ProgressBar.ProgressBar(i, frames, "", time_index=clock_index)
        c=fd[:,i,0];
        flow = Reconstruct_Velocity(c.flatten(), None, True, res=resolution)
        visualize_velocity(flow, x1, x2, framedata+ending.format(count_i), use_liq=True)
        count_i = count_i + 1
    ProgressBar.ProgressBar(frames, frames, "", time_index=clock_index)
    video = result_directory + video_name
    input = framedata + "%0"+str(l+1)+"d.png"
    to_execute="yes | ffmpeg -r {} -start_number_range {} -f image2 -i {} -vcodec libx264 -pix_fmt yuv420p {}".format(out_fps, count_i, input, video)
    PrintInfo(to_execute)
    os.system(to_execute)


def main(args):
    Render(args)

if __name__ == "__main__":
    main(sys.argv[1:])
