import numpy as np

def Project_Vorticity(samples , number_basis_functions ):
    # get the resolution of the samples
    res_x = samples.shape[0];
    res_y = samples.shape[1];

    # expand the signal to the range [0, 2pi]
    s = np.zeros([(res_x-1)*2, (res_y-1)*2]);
    s[0:res_x, 0:res_y] = samples;
    s[res_x: , 0:res_y] = -samples[-2:0:-1,  :    ];
    s[0:res_x, res_y: ] = -samples[  :   , -2:0:-1];
    s[res_x: , res_y: ] = samples[-2:0:-1, -2:0:-1];

    # project the vorticity using the fft
    tmp = np.imag(np.fft.fft(np.imag(np.fft.fft(s, axis=0)), axis=1))
    # transpose the coefficient to fit the use in the program
    tmp = np.transpose(tmp);
    # scale it down by the amount of possible frequencies
    tmp *= 1/((s.shape[0]/2)*(s.shape[1]/2))
    # discard all coefficients of basisfunctions not corresbonding to the given ones
    tmp = tmp[0:number_basis_functions+1, 0:number_basis_functions+1];

    # return the coefficients as an array
    return tmp.reshape((number_basis_functions+1)**2);

def Project_Velocity(samples_x , samples_y , number_basis_functions ):
    # getting the resolution of the samples
    res_x = samples_x.shape[0];
    res_y = samples_y.shape[1];

    # expand the signal to the range [0, 2pi]
    s_x = np.zeros([(res_x-1)*2, (res_y-1)*2]);
    s_x[0:res_x, 0:res_y] = samples_x;
    s_x[res_x: , 0:res_y] = samples_x[-2:0:-1,   :    ];
    s_x[0:res_x, res_y: ] =-samples_x[  :    , -2:0:-1];
    s_x[res_x: , res_y: ] =-samples_x[-2:0:-1, -2:0:-1];

    s_y = np.zeros([(res_x-1)*2, (res_y-1)*2]);
    s_y[0:res_x, 0:res_y] = samples_y;
    s_y[res_x: , 0:res_y] =-samples_y[-2:0:-1,   :    ];
    s_y[0:res_x, res_y: ] = samples_y[  :    , -2:0:-1];
    s_y[res_x: , res_y: ] =-samples_y[-2:0:-1, -2:0:-1];

    # using the fft extract the coefficients from the signals
    # also scale the fft factor out
    tmp_x = np.real(np.fft.fft(np.imag(np.fft.fft(s_x, axis=1)), axis=0))/((res_x-1)*(res_y-1))
    tmp_y = -np.imag(np.fft.fft(np.real(np.fft.fft(s_y, axis=1)), axis=0))/((res_x-1)*(res_y-1))

    # discard all irrelevant frequencies
    tmp_x = tmp_x[0:number_basis_functions+1, 0:number_basis_functions+1];
    tmp_y = tmp_y[0:number_basis_functions+1, 0:number_basis_functions+1];

    # getting the frequencies which corresponds to the coefficients
    k1 = range(0, number_basis_functions+1);
    k2 = range(0, number_basis_functions+1);

    k1, k2 = np.meshgrid(k1, k2);
    lambda_k = k1*k1 + k2*k2;

    # scale up by the eigenvalues of the functions
    tmp_x *= lambda_k;
    tmp_y *= lambda_k;

    # avoid deviding by zero but devide by the correct frequencies
    tmp_x[1:res_x, :] = tmp_x[1:res_x, :]/k2[1:res_x, :];
    tmp_y[:,1:res_y] = tmp_y[:,1:res_y]/k1[:,1:res_y];

    # the frequencies are negative for some reason. Fix by multiplying with -1
    tmp_x = -tmp_x;
    tmp_y = -tmp_y;

    return np.transpose(tmp_x), np.transpose(tmp_y);
