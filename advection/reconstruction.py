import numpy as np

def Reconstruct_Vorticity(coeffs , res_x , res_y , res_modifier = 1):
    number_basis_functions = int(np.sqrt(len(coeffs)))-1;
    # put coefficients in matrix corresponding to the frequencies
    c = coeffs.reshape((number_basis_functions+1), (number_basis_functions+1));

    # enlarge the coefficient array by padding with zeros
    # the new coefficient array must have the resolution size of the reconstruction
    working_c = np.zeros([4*res_modifier*(res_x-1), 4*res_modifier*(res_y-1)]);

    # adding the missing coefficients corresponding to the negative frequencies
    working_c[0:number_basis_functions+1, 0:number_basis_functions+1] = c;
    working_c[-1:-number_basis_functions-1:-1, 0:number_basis_functions+1] = -c[1:number_basis_functions+1, :];
    working_c[0:number_basis_functions+1, -1:-number_basis_functions-1:-1] = -c[:,1:number_basis_functions+1];
    working_c[-1:-number_basis_functions-1:-1, -1:-number_basis_functions-1:-1]=c[1:number_basis_functions+1, 1:number_basis_functions+1];

    # scale by the amount of frequencies
    working_c *= (2*res_modifier)**2*(res_x-1)*(res_y-1);

    # in code the x and y coefficients are treated transposed # TODO: fix this issue
    working_c = np.transpose(working_c);

    # reconstructing the vorticity by using the fourier transformation
    res = np.imag(np.fft.ifft(np.imag(np.fft.ifft(working_c, axis=0)), axis=1));
    # picking the right sampling points and discarding everything past pi
    return res[0:2*res_modifier*(res_x-1)+1:2, 0:2*res_modifier*(res_y-1)+1:2];

def GetRecCoeffs(coeffs , type = 0):
    c = np.copy(coeffs);

    # getting the frequencies which corresponds to the coefficients
    k1 = range(0, c.shape[1]);
    k2 = range(0, c.shape[0]);

    k2, k1 = np.meshgrid(k1, k2);
    lambda_k = k1*k1 + k2*k2;

    # depending on if it is the x-component or y-component it needs to be scaled differently
    if type == 0:
        c *= k2;
    else:
        c *= -k1;

    # avoid deviding by zero but still scale the parameter down
    c[1:, 1:] = c[1:, 1:] / lambda_k[1:, 1:];

    return c;

def Reconstruct_Velocity(coeffs , basisfunctions , usefft = False, res_modifier = 1, return_2pi = False, res = None):
    if basisfunctions is None and usefft == False:
        raise Exception("Expected basisfunctions to be given");
    elif basisfunctions is None and res is None:
        raise Exception("resolution must be defined");
    elif basisfunctions is None:
        numsquared = coeffs.shape[0];
        two = 2;
        lenx = res[0];
        leny = res[1];
    else:
        numsquared, two, lenx, leny = basisfunctions.shape;
    if two != 2:
        raise Exception("Bad dimensions");

    flow = np.zeros([two, res_modifier*(lenx-1)+1, res_modifier*(leny-1)+1]);

    if usefft:
        # reshape coefficients into a matrix with entries c[k1, k2] = xi_k
        c = np.reshape(coeffs, ((int)(np.sqrt(numsquared)), (int)(np.sqrt(numsquared))));
        # extract number of basis functions from the shape of that matrix
        n_b = c.shape[0]-1;

        # scale coefficients according to the basis functions
        vx_ = np.transpose(GetRecCoeffs(c, 0));
        vy_ = np.transpose(GetRecCoeffs(c, 1));

        # enlarge the coefficient arrays to match the resolution and to fit the coefficients
        # of the negative frequencies
        vx = np.zeros([4*res_modifier*(lenx-1), 4*res_modifier*(leny-1)]);
        vy = np.zeros([4*res_modifier*(lenx-1), 4*res_modifier*(leny-1)]);

        # fill in the negative frequency coefficients
        vx[0:n_b+1, 0:n_b+1] = vx_;
        vx[-1:-n_b-1:-1, 0:n_b+1] = vx_[1:n_b+1, :];
        vx[0:n_b+1, -1:-n_b-1:-1] = -vx_[:, 1:n_b+1];
        vx[-1:-n_b-1:-1, -1:-n_b-1:-1] = -vx_[1:n_b+1, 1:n_b+1];

        vy[0:n_b+1, 0:n_b+1] = vy_;
        vy[-1:-n_b-1:-1, 0:n_b+1] = -vy_[1:n_b+1, :];
        vy[0:n_b+1, -1:-n_b-1:-1] = vy_[:, 1:n_b+1];
        vy[-1:-n_b-1:-1, -1:-n_b-1:-1] = -vy_[1:n_b+1, 1:n_b+1];

        # scale the coefficients for the fft
        vx *= (2*res_modifier)**2*(lenx-1)*(leny-1);
        vy *= (2*res_modifier)**2*(lenx-1)*(leny-1);

        # reconstruct the signal in the range [0, 2*pi]
        vx = np.imag(np.fft.ifft(np.real(np.fft.ifft(vx, axis= 0)), axis= 1));
        vy = np.real(np.fft.ifft(np.imag(np.fft.ifft(vy, axis= 0)), axis= 1));

        if return_2pi:
            return np.array([vx, vy])

        # pick the resolution many sample points in range [0,pi]
        flow[0,:,:] = vx[0:2*res_modifier*(lenx-1)+1:2, 0:2*res_modifier*(leny-1)+1:2];
        flow[1,:,:] = vy[0:2*res_modifier*(lenx-1)+1:2, 0:2*res_modifier*(leny-1)+1:2];
    else:
        for i in range(numsquared):
            flow += coeffs[i]*basisfunctions[i];
    return flow;
