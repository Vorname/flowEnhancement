import numpy as np

N=100;

# freq = np.fft.fftfreq(N, 1/N);
#
# t = np.zeros(N);
# t[1] = 50;
# t[-1] = -50;
#
# test = np.imag(np.fft.ifft(t));
# sin = np.sin([2*np.pi*n/N for n in range(N)]);
#
# print(test-sin)

sinsin = np.zeros([N, N]);
for i in range(N):
    for j in range(N):
        sinsin[i,j] = np.sin(2*np.pi*i/N)*np.sin(2*np.pi*j/N);

print(np.real(np.fft.fft(np.real(np.fft.fft(sinsin, axis=0)), axis=1)));

params = np.zeros([N, N]);
params[1,1] = (N/2)**2;
params[-1,1] = -params[1,1];
params[1,-1] = -params[1,1];
params[-1,-1] = params[1,1];

reconstruct = np.real(np.fft.ifft(np.real(np.fft.ifft(params, axis=0)), axis=1));

print(np.linalg.norm(sinsin.reshape(N*N) - reconstruct.reshape(N*N)));
