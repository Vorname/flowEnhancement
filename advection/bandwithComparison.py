from flow_data import FlowData
import numpy as np
from Log import *
from integration import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# currently unused

class BandwithData:
    __basis_M = None
    __basis_N = None
    __basis_inital = None
    __viscosity = 0
    __t_end = None
    __fps = None
    __num_iterations = None
    __coeff_N_data = None
    __coeff_M_data = None
    __energy_data = None
    __initial_flow = None


    def __del__(self):
        print("deleted object successfully")

    def get_N(self):
        return self.__coeff_N_data

    def get_M(self):
        return self.__coeff_M_data

    def __init__(self, basis_M,basis_N, t_end,fps, num_iterations, initialFlowData = None, normalizeInitialFlow = False, viscosity = 0, basis_initial = None, printProgress=False, epsilon = 0, useEuler = False):
        self.__basis_M = basis_M
        self.__basis_N = basis_N
        self.__viscosity = viscosity
        self.__t_end = t_end
        self.__fps = fps
        self.__num_iterations = num_iterations
        self.__coeff_N_data = np.zeros([(basis_N+1)**2, int(t_end), num_iterations])
        self.__coeff_M_data = np.zeros([(basis_M+1)**2, int(t_end), num_iterations])
        self.__energy_data = np.zeros([2,int(t_end*fps),num_iterations])
        self.printProgress = printProgress
        self.epsilon = epsilon
        self.useEuler = useEuler

        if basis_initial is None:
            self.__basis_initial = basis_M
        else:
            self.__basis_inital = basis_initial

        if initialFlowData is None:
            self.__initial_flow = np.zeros([(basis_N+1)*(basis_N+1),num_iterations])
            for i in range(0,num_iterations):
                random_flow = np.zeros([self.__basis_N+1, self.__basis_N+1])
                random_flow[1:basis_initial+1, 1:basis_initial+1] = np.random.rand(basis_initial, basis_initial)-0.5
                random_flow = random_flow.reshape((self.__basis_N+1)*(self.__basis_N+1))
                if normalizeInitialFlow is True:
                    for i in range(1, self.__basis_N+1):
                        for j in range(1, self.__basis_N+1):
                            random_flow[i*self.__basis_N+j] = 1/-lambda_k(i,j)*random_flow[i*self.__basis_N+j]
                self.__initial_flow[:,i] = random_flow
        else:
            self.__initial_flow = initialFlowData
        
        if self.epsilon is not None:
            for i in range(0,num_iterations):
                self.__initial_flow[:,i] = self.__initial_flow[:,0]
        else:
            self.epsilon = 0

    def ComputeData(self, normalizeEpsilonFlow = True, use_fft = True):

        time_index_global = StartClock()
        time_index = StartClock()

        if (not use_fft):
            # Precompute advection matrices
            adv_M = BuildAdvectionMatrix(self.__basis_M, printProgress=True, time_index=time_index)
            adv_N = BuildAdvectionMatrix(self.__basis_N, printProgress=True, time_index=time_index)
        else:
            adv_M = None
            adv_N = None
        
        for step in range(self.__num_iterations):
            flow_N = self.__initial_flow[:,step]
            random_flow = np.zeros([self.__basis_N+1, self.__basis_N+1])
            random_flow[1:self.__basis_initial+1, 1:self.__basis_initial+1] = np.random.randn(self.__basis_initial, self.__basis_initial)-0.5
            random_flow = random_flow.reshape((self.__basis_N+1)*(self.__basis_N+1))

            if normalizeEpsilonFlow is True:
                for i in range(1, self.__basis_N+1):
                    for j in range(1, self.__basis_N+1):
                        random_flow[i*self.__basis_N+j] = 1/-lambda_k(i,j)*random_flow[i*self.__basis_N+j]

            # offset flow by epsilon into neighboring region
            if step != 0:
                flow_N = flow_N + self.epsilon * random_flow
            #print(flow_N)
            # extract M coeffs
            flow_M = flow_N.reshape([self.__basis_N+1,-1])[:self.__basis_M+1,:self.__basis_M+1]
            flow_M = flow_M.reshape((self.__basis_M+1)*(self.__basis_M+1))
            print("Debug: {}, {}, {}, {}, {}\n".format(adv_N, self.__viscosity, self.__t_end, self.__fps, np.shape( self.__coeff_N_data[:,:,step]) ))
            if self.__num_iterations > 1:
                IntegrateSimple(flow_N,self.__viscosity,self.__t_end,self.__fps, energy_data=self.__energy_data[0,:,step], coeff_data = self.__coeff_N_data[:,:,step], useEuler=self.useEuler)
                IntegrateSimple(flow_M,self.__viscosity,self.__t_end,self.__fps, energy_data=self.__energy_data[1,:,step], coeff_data = self.__coeff_M_data[:,:,step], useEuler=self.useEuler)
                ReStartClock(time_index)
                ProgressBar.ProgressBar(step, self.__num_iterations, "Bandwith Data beenig computed...", time_index=time_index)
            else:
                ReStartClock(time_index)
                IntegrateSimple(flow_N,self.__viscosity,self.__t_end,self.__fps, energy_data=self.__energy_data[0,:,step], coeff_data = self.__coeff_N_data[:,:,step], useEuler=self.useEuler)
                IntegrateSimple(flow_M,self.__viscosity,self.__t_end,self.__fps, energy_data=self.__energy_data[1,:,step], coeff_data = self.__coeff_M_data[:,:,step], useEuler=self.useEuler)
               
        PrintInfo("Computation of bandwith data [{}x{}x{}] took {}s \n".format(self.__basis_M,self.__basis_N,self.__num_iterations,GetTime(time_index_global)))


    #def PlotCoeffs(self, plotEngergy = True, plotAll = False):

    def PlotError(self):

        # store min, max, mean, median for each timeframe and coeff
        data = np.zeros([(self.__basis_M+1)**2, int(self.__t_end * self.__fps), 4])

        for i in range(0,self.__basis_M+1):
            for j in range(0,self.__basis_M+1):
                data[i*(self.__basis_M+1)+j, :, 0] = np.min(    np.absolute(np.subtract(self.__coeff_M_data[i*(self.__basis_M+1)+j,:,:], self.__coeff_N_data[i*(self.__basis_N+1)+j,:,:])),axis=1)
                data[i*(self.__basis_M+1)+j, :, 1] = np.max(    np.absolute(np.subtract(self.__coeff_M_data[i*(self.__basis_M+1)+j,:,:], self.__coeff_N_data[i*(self.__basis_N+1)+j,:,:])),axis=1)
                data[i*(self.__basis_M+1)+j, :, 2] = np.mean(   np.absolute(np.subtract(self.__coeff_M_data[i*(self.__basis_M+1)+j,:,:], self.__coeff_N_data[i*(self.__basis_N+1)+j,:,:])),axis=1)
                data[i*(self.__basis_M+1)+j, :, 3] = np.median( np.absolute(np.subtract(self.__coeff_M_data[i*(self.__basis_M+1)+j,:,:], self.__coeff_N_data[i*(self.__basis_N+1)+j,:,:])),axis=1)

        x_axis_data = [float(i)/self.__fps for i in range(int(self.__t_end*self.__fps))]

        fig = make_subplots(rows=self.__basis_M, cols=self.__basis_M, start_cell="top-left")

        for i in range(1,self.__basis_M+1):
            for j in range(1,self.__basis_M+1):
                fig.append_trace(go.Scatter(
                x=x_axis_data,
                y=data[i*(self.__basis_M+1)+j,:,0],
                mode='lines',
                line=dict(color='blue'),
                fill = 'none',
                name = "N: K({},{})".format(i,j)
                ), row=i, col=j)

                fig.append_trace(go.Scatter(
                x=x_axis_data,
                y=data[i*(self.__basis_M+1)+j,:,1],
                mode='lines',
                line=dict(color='blue'),
                fill = 'tonexty',
                name = "N: K({},{})".format(i,j)
                ), row=i, col=j)

                fig.append_trace(go.Scatter(
                x=x_axis_data,
                y=data[i*(self.__basis_M+1)+j,:,2],
                mode='lines',
                line=dict(color='red'),
                fill = 'none',
                name = "N: K({},{})".format(i,j)
                ), row=i, col=j)

                fig.append_trace(go.Scatter(
                x=x_axis_data,
                y=data[i*(self.__basis_M+1)+j,:,3],
                mode='lines',
                line=dict(color='black'),
                fill = 'none',
                name = "N: K({},{})".format(i,j)
                ), row=i, col=j)
        
        return(fig)

    def PlotEnergy(self):

        data = np.zeros([int(self.__t_end * self.__fps), 4])
        data[:, 0] = np.min(    np.absolute(np.subtract(self.__energy_data[0,:,:], self.__energy_data[1,:,:])),axis=1)
        data[:, 1] = np.max(    np.absolute(np.subtract(self.__energy_data[0,:,:], self.__energy_data[1,:,:])),axis=1)
        data[:, 2] = np.mean(   np.absolute(np.subtract(self.__energy_data[0,:,:], self.__energy_data[1,:,:])),axis=1)
        data[:, 3] = np.median( np.absolute(np.subtract(self.__energy_data[0,:,:], self.__energy_data[1,:,:])),axis=1)

        x_axis_data = [float(i)/self.__fps for i in range(int(self.__t_end*self.__fps))]

        fig = make_subplots(rows=1, cols=1, start_cell="top-left")
        fig.append_trace(go.Scatter(
            x=x_axis_data,
            y=data[:,0],
            mode='lines',
            line=dict(color='blue'),
            fill='none'
        ),row=1,col=1)
        fig.append_trace(go.Scatter(
            x=x_axis_data,
            y=data[:,1],
            mode='lines',
            line=dict(color='blue'),
            fill='tonexty'
        ),row=1,col=1)        
        fig.append_trace(go.Scatter(
            x=x_axis_data,
            y=data[:,2],
            mode='lines',
            line=dict(color='red'),
            fill='none'
        ),row=1,col=1)        
        fig.append_trace(go.Scatter(
            x=x_axis_data,
            y=data[:,3],
            mode='lines',
            line=dict(color='black'),
            fill='none'
        ),row=1,col=1)
        return(fig)

    def PlotSingleSlice(self,slice_index,show_Engery = False, print_Full = True, filterZero = True, share_x=False, share_y=True):
        
        x_axis_data = [float(i)/self.__fps for i in range(int(self.__t_end*self.__fps))]

        num_rows = self.__basis_M
        num_cols = self.__basis_M
        if (print_Full):
            num_rows = self.__basis_N
            num_cols = self.__basis_N


        fig_N = make_subplots(rows=num_rows + (int)(show_Engery==True), cols=num_cols, start_cell="top-left",subplot_titles = ("K1",""),shared_xaxes=share_x,shared_yaxes=share_y)

        # force axis to be shared row wise using fake data points
        min_val = np.amin([np.amin(self.__coeff_M_data[:,:,slice_index]),np.amin(self.__coeff_N_data[:,:,slice_index])])
        max_val = np.amax([np.amax(self.__coeff_M_data[:,:,slice_index]),np.amax(self.__coeff_N_data[:,:,slice_index])])
        max_x = np.amax(x_axis_data)

        # round to set very small values zero
        if (filterZero):
            pres = 15
        else:
            pres = 100

        for i in range(1,num_rows + 1):
            for j in range(1,num_cols + 1):
                if (i <= self.__basis_M and j <= self.__basis_M):
                    fig_N.append_trace(go.Scatter(
                    x=x_axis_data,
                    y=np.round(self.__coeff_M_data[i*(self.__basis_M+1)+j,:,slice_index],pres),
                    yaxis = 'y',
                    mode='lines',
                    line=dict(color='blue'),
                    fill = 'none',
                    name = "M: K({},{})".format(i,j)
                    ), row=i, col=j)

                fig_N.append_trace(go.Scatter(
                x=x_axis_data,
                y=np.round(self.__coeff_N_data[i*(self.__basis_N+1)+j,:,slice_index],pres),
                yaxis = 'y',
                mode='lines',
                line=dict(color='black'),
                fill = 'none',
                name = "N: K({},{})".format(i,j)
                ), row=i, col=j)
                # set min and max in last column for every subplot to force equal axis along rows
                if (j == self.__basis_M):
                    fig_N.append_trace(go.Scatter(
                    x = [max_x],
                    y = [min_val],
                    mode='markers',
                    opacity=0,
                    ), row=i, col=j)
                    fig_N.append_trace(go.Scatter(
                    x = [max_x],
                    y = [max_val],
                    mode='markers',
                    opacity=0,
                    ), row=i, col=j)
                
        if (show_Engery):
            fig_N.append_trace(go.Scatter(
            x=x_axis_data,
            y=self.__energy_data[0,:,slice_index],
            mode='lines',
            line=dict(color='blue'),
            fill = 'none',
            name = "M-Energy"
            ), row=num_rows + 1, col=1)

            fig_N.append_trace(go.Scatter(
            x=x_axis_data,
            y=self.__energy_data[1,:,slice_index],
            mode='lines',
            line=dict(color='black'),
            fill = 'none',
            name = "N-Energy)"
            ), row=num_rows + 1, col=1)
        fig_N.update_layout(title_text="M = {}, N = {}, timesteps = {}, fps = {}".format(self.__basis_M,self.__basis_N,self.__t_end,self.__fps))
        fig_N.update_yaxes(title_text="K2", row=1, col=1)

        fig_M = None
        if print_Full:
            fig_M, _ = self.PlotSingleSlice(slice_index,print_Full=False)

        return fig_N, fig_M

    def __PlotMultipleSlice(self, show_Engery = False, print_Full = True, filterZero = True, share_x=False, share_y=True, forceAxisEqual = True):
        
        x_axis_data = [i for i in range(int(self.__t_end*self.__fps))]

        num_rows = self.__basis_M
        num_cols = self.__basis_M
        if (print_Full):
            num_rows = self.__basis_N
            num_cols = self.__basis_N


        fig = make_subplots(rows=num_rows + (int)(show_Engery==True), cols=num_cols, start_cell="top-left",subplot_titles = ("K1",""),shared_xaxes=share_x,shared_yaxes=share_y)

        # force axis to be shared row wise using fake data points
        min_val = np.amin([np.amin(self.__coeff_M_data[:,:,:]),np.amin(self.__coeff_N_data[:,:,:])])
        max_val = np.amax([np.amax(self.__coeff_M_data[:,:,:]),np.amax(self.__coeff_N_data[:,:,:])])
        max_x = np.amax(x_axis_data)

        # round to set very small values zero
        if (filterZero):
            pres = 15
        else:
            pres = 100


        for slice_index in range(0,self.__num_iterations):
            for i in range(1,num_rows + 1):
                for j in range(1,num_cols + 1):
                    if (i <= self.__basis_M and j <= self.__basis_M):
                        fig.append_trace(go.Scatter(
                        x=x_axis_data,
                        y=np.round(self.__coeff_M_data[i*(self.__basis_M+1)+j,:,slice_index],pres),
                        yaxis = 'y',
                        mode='lines',
                        line=dict(color='black'),
                        fill = 'none',
                        name = "M: K({},{})".format(i,j)
                        ), row=i, col=j)

                    fig.append_trace(go.Scatter(
                    x=x_axis_data,
                    y=np.round(self.__coeff_N_data[i*(self.__basis_N+1)+j,:,slice_index],pres),
                    yaxis = 'y',
                    mode='lines',
                    line=dict(color='blue'),
                    fill = 'none',
                    name = "N: K({},{})".format(i,j)
                    ), row=i, col=j)
                    # set min and max in last column for every subplot to force equal axis along rows
                    if (j == self.__basis_M and forceAxisEqual):
                        fig.append_trace(go.Scatter(
                        x = [max_x],
                        y = [min_val],
                        mode='markers',
                        opacity=0,
                        ), row=i, col=j)
                        fig.append_trace(go.Scatter(
                        x = [max_x],
                        y = [max_val],
                        mode='markers',
                        opacity=0,
                        ), row=i, col=j)
                    
            if (show_Engery):
                fig.append_trace(go.Scatter(
                x=x_axis_data,
                y=self.__energy_data[0,:,slice_index],
                mode='lines',
                line=dict(color='black'),
                fill = 'none',
                name = "M-Energy"
                ), row=num_rows + 1, col=1)

                fig.append_trace(go.Scatter(
                x=x_axis_data,
                y=self.__energy_data[1,:,slice_index],
                mode='lines',
                line=dict(color='blue'),
                fill = 'none',
                name = "N-Energy)"
                ), row=num_rows + 1, col=1)
        fig.update_layout(showlegend=False,title_text="M = {}, N = {}, timesteps = {}, fps = {}, visc = {}, eps = {}, num_runs = {}".format(self.__basis_M,self.__basis_N,self.__t_end,self.__fps,self.__viscosity, self.epsilon, self.__num_iterations))
        fig.update_yaxes(title_text="K2", row=1, col=1)

        return fig

    def PlotMultipleSlice(self, show_Engery = False, filterZero = True, share_x=False, share_y=True, forceAxisEqual = True):
        p_M = self.__PlotMultipleSlice(show_Engery = show_Engery, print_Full = False, filterZero = filterZero, share_x=share_x, share_y=share_y, forceAxisEqual=forceAxisEqual)
        p_N = self.__PlotMultipleSlice(show_Engery = show_Engery, print_Full = True, filterZero = filterZero, share_x=share_x, share_y=share_y, forceAxisEqual=forceAxisEqual)
        return p_M, p_N

    def PlotDistance(self, numLines=10):
        # compute difference to base flow
        M_dist = np.zeros([int(self.__t_end), self.__num_iterations])
        N_dist = np.zeros([int(self.__t_end), self.__num_iterations])
        for t in range(self.__t_end):
            for k in range(self.__num_iterations):
                M_dist[t,k] = np.sum((self.__coeff_M_data[:,t,k]-self.__coeff_M_data[:,t,0])**2)
                N_dist[t,k] = np.sum((self.__coeff_M_data[:,t,k]-self.__coeff_M_data[:,t,0])**2)
        fig = go.Figure()
        for i in range(0, self.__num_iterations, max(1,int(self.__num_iterations/numLines))):
            fig.add_trace(go.Scatter(y=M_dist[:,k], name = 'M distance',line = dict(color='firebrick')))
            fig.add_trace(go.Scatter(y=N_dist[:,k], name = 'N distance',line = dict(color='royalblue')))
        fig.add_trace(go.Scatter(y=np.min(M_dist,axis=1), name = 'M distance min',line = dict(color='firebrick',dash='dash')))
        fig.add_trace(go.Scatter(y=np.mean(M_dist,axis=1), name = 'M distance mean',line = dict(color='firebrick')))
        fig.add_trace(go.Scatter(y=np.max(M_dist,axis=1), name = 'M distance max',line = dict(color='firebrick',dash='dash')))
        fig.add_trace(go.Scatter(y=np.min(N_dist,axis=1), name = 'N distance min',line = dict(color='royalblue',dash='dash')))
        fig.add_trace(go.Scatter(y=np.mean(N_dist,axis=1), name = 'N distance mean',line = dict(color='royalblue')))
        fig.add_trace(go.Scatter(y=np.max(N_dist,axis=1), name = 'N distance max',line = dict(color='royalblue',dash='dash')))

        return fig
        






