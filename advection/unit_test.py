import numpy as np
import integration as integ
import projection as proj
import reconstruction as reco
import visualize_flow as visual
from global_functions import *
import time
from Log import ProgressBar as progress
from Log import Logger as log
from Log import PrintInfo
import itertools as iter

np.random.seed(1024509123)

def MatrixToString(mat):
    res = "[";
    max_length = 0;
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if(len(str(mat[i,j])) > max_length):
                max_length = len(str(mat[i,j]));
    for i in range(mat.shape[0]):
        res += "["
        for j in range(mat.shape[1]):
            tmp = str(mat[i,j]);
            for k in range(len(tmp), max_length):
                tmp += " ";
            res += tmp + " ";
        res += "]\n";

    return res + "]";

number_basis_functions = 3;
grid = 10;
test_runs = 10;

PrintInfo("Initializing Unit test with {:01} basis functions for each frequence, {:01} grid resolution for reconstructions and {:01} test runs".format(number_basis_functions, grid, test_runs));
x = np.linspace(0, np.pi, grid);
y = np.linspace(0, np.pi, grid);
x1, x2 = np.meshgrid(x,y);

start = time.perf_counter();
basisfunctions_velocity = GetBasisfunctions(number_basis_functions, x1, x2);
basisfunctions_vorticity = GetBasis_vortisity_functions(number_basis_functions, x1, x2);
end = time.perf_counter();

errors_reconstruction = np.zeros([test_runs, 3]);
errors_projection = np.zeros([test_runs, 3]);

reconstruction_time_reference = [];

reconstruction_time = [];
projection_times = [];


advection_errors=np.zeros([test_runs, 2]);

visualize = False;
initializing_time = (end-start)*1000
PrintInfo("Initializing complete. It took {}ms".format((end-start)*1000));

advection_reference = BuildAdvectionMatrix(number_basis_functions);
advection_compare = BuildAdvectionMatrix(number_basis_functions, usefft = True);

error_advection = (advection_compare-advection_reference).Norm();
progress.PrintTestResult(error_advection < 0.001, "Advection Test", "advection error = {}".format(error_advection));

anti_symetric = True;
anti_symetric_reference = True;
def inverse_lambda(i):
    global number_basis_functions
    i1 = int(i/(number_basis_functions+1));
    i2 = int(i%(number_basis_functions+1));
    if i1 == 0 or i2 == 0:
        return 0;
    return 1/lambda_k(i1, i2);

anti_symetric_errors = [];
anti_symetric_errors_reference = [];
for i, j, k in iter.product(range(advection_compare.shape[0]), range(advection_compare.shape[1]), range(advection_compare.shape[2])):
    inv_j = inverse_lambda(j);
    inv_k = inverse_lambda(k);
    err   = inv_j*advection_compare[i,j,k] + inv_k*advection_compare[i,k,j]
    err_r = inv_j*advection_reference[i,j,k] + inv_k*advection_reference[i,k,j]
    anti_symetric_errors.append(err);
    anti_symetric_errors_reference.append(err_r);
    if np.abs(err) > 0.001:
        anti_symetric = False;
    if np.abs(err_r) > 0.001:
        anti_symetric_reference = False;

progress.PrintTestResult(anti_symetric, "Anti-symetrie Test", "The advection matrix build using fft is not anti symetric, average error = {}; max_error = {}".format(np.sum(np.abs(anti_symetric_errors))/len(anti_symetric_errors), np.max(np.abs(anti_symetric_errors))));
progress.PrintTestResult(anti_symetric_reference, "Reference is anti symetric", "The advection matrix should be anti symetric, average error = {}; max_error = {}".format(np.sum(np.abs(anti_symetric_errors_reference))/len(anti_symetric_errors_reference), np.max(np.abs(anti_symetric_errors_reference))));


print("");
for i in range(test_runs):
    progress.ProgressBar(i, test_runs, "Running Tests");
    random_flow_coeffs = np.zeros([number_basis_functions+1, number_basis_functions+1]);
    random_flow_coeffs[1:number_basis_functions+1, 1:number_basis_functions+1] = np.random.randn(number_basis_functions, number_basis_functions);
    random_flow_coeffs = random_flow_coeffs.reshape((number_basis_functions+1)*(number_basis_functions+1))

    #Does reconstruction work?
    start = time.perf_counter();
    reference_velocity = reco.Reconstruct_Velocity(random_flow_coeffs, basisfunctions_velocity, usefft = False);
    end_reference = time.perf_counter();
    comparison_velocity = reco.Reconstruct_Velocity(random_flow_coeffs, basisfunctions_velocity, usefft = True);
    end = time.perf_counter();

    reconstruction_time_reference.append(end_reference - start);
    reconstruction_time.append(end - end_reference);

    reference_vorticity = np.zeros([grid, grid]);
    for j in range(random_flow_coeffs.shape[0]):
        reference_vorticity += random_flow_coeffs[j]*basisfunctions_vorticity[j];
    comparison_vorticity = reco.Reconstruct_Vorticity(random_flow_coeffs, grid, grid);

    if visualize:
        visual.visualize_velocity(reference_velocity, x1, x2, "../data/images/tmp/{:02}_reference_velocity".format(i), use_liq = False);
        visual.visualize_velocity(comparison_velocity, x1, x2, "../data/images/tmp/{:02}_comparison_velocity".format(i), use_liq = False);
        visual.visualize_vorticity(reference_vorticity, x1, x2, "../data/images/tmp/{:02}_reference_vorticity".format(i));
        visual.visualize_vorticity(comparison_vorticity, x1, x2, "../data/images/tmp/{:02}_comparison_vorticity".format(i));

    difference = reference_velocity-comparison_velocity;
    diff_vorticity = reference_vorticity-comparison_vorticity;
    errors_reconstruction[i, 0] = np.linalg.norm(difference[0,:,:]);
    errors_reconstruction[i, 1] = np.linalg.norm(difference[1,:,:]);
    errors_reconstruction[i, 2] = np.linalg.norm(diff_vorticity);
    ####################################################################
    #Does Projection work?
    vorticity_double_res = reco.Reconstruct_Vorticity(random_flow_coeffs, grid, grid, res_modifier = 2);
    projection_vorticity = proj.Project_Vorticity(vorticity_double_res, number_basis_functions);

    velocity_double_res = reco.Reconstruct_Velocity(random_flow_coeffs, basisfunctions_velocity, usefft = True, res_modifier = 2);
    start = time.perf_counter();
    x_coeffs, y_coeffs = proj.Project_Velocity(velocity_double_res[0], velocity_double_res[1], number_basis_functions);
    end = time.perf_counter();
    x_coeffs = x_coeffs.reshape((number_basis_functions+1)**2);
    y_coeffs = y_coeffs.reshape((number_basis_functions+1)**2);

    projection_times.append(end - start);

    difference = projection_vorticity - random_flow_coeffs;
    errors_projection[i, 0] = np.linalg.norm(random_flow_coeffs - x_coeffs);
    errors_projection[i, 1] = np.linalg.norm(random_flow_coeffs - y_coeffs);
    errors_projection[i, 2] = np.linalg.norm(difference)
    ######################################################################
    #Does Advection work
    adv = advection_compare.Apply(random_flow_coeffs);
    adv_p, adv_n = Advection(random_flow_coeffs, advection_compare, 0, usefft=False);
    adv_p_mf, adv_n_mf = Advection(random_flow_coeffs, advection_compare, 0, usefft=True);
    if i == 1:
        Logger.Log(advection_compare.Apply(random_flow_coeffs))

    advection_errors[i, 0] = np.linalg.norm(adv-adv_p);
    advection_errors[i, 1] = np.linalg.norm(adv-adv_p_mf);

progress.ProgressBar(test_runs, test_runs, "Running Tests")

Passed_Reconstruction = True;
Passed_Projection = True;
Advection_correct = True;
Advection_Matrixfree=True;
for i in range(test_runs):
    x_r, y_r, v_r = np.abs(errors_reconstruction[i]);
    x_p, y_p, v_p = np.abs(errors_projection[i]);
    a_a, a_mf = np.abs(advection_errors[i]);
    if x_r > np.finfo(float).eps * 30 * grid or y_r > np.finfo(float).eps * 30*grid or v_r > np.finfo(float).eps * 30*grid:
        Passed_Reconstruction = False;
    if x_p > np.finfo(float).eps * 30*grid or y_p > np.finfo(float).eps * 30*grid or v_p > np.finfo(float).eps * 30*grid:
        Passed_Projection = False;
    tol = np.finfo(float).eps *((number_basis_functions+1)**2 + 2*((number_basis_functions+1)**2)**2)*grid;
    if a_a > tol:
        Advection_correct = False;
    if a_mf > tol:
        Advection_Matrixfree = False;

progress.PrintTestResult(Passed_Reconstruction, "Reconstruction Test", "max error={} tolerance was {}".format(np.max(np.abs(errors_reconstruction)), np.finfo(float).eps * 30*grid));
progress.PrintTestResult(Passed_Projection, "Projection Test", "max error={} tolerance was {}".format(np.max(np.abs(errors_projection)), np.finfo(float).eps * 30*grid));
progress.PrintTestResult(Advection_correct, "The preserving part of Advection is the application of matrix", "max error = {} tolerance = {}".format(np.max(advection_errors[:,0]), tol));
progress.PrintTestResult(Advection_Matrixfree, "The matrix free Advection is correct", "max error = {} tolerance = {}".format(np.max(advection_errors[:,1]), tol));

PrintInfo("average reference times = {}ms\nreconstruction times = {}ms\nprojection times = {}ms".format(np.sum(reconstruction_time_reference)*1000/len(reconstruction_time_reference) + initializing_time, np.sum(reconstruction_time)*1000/len(reconstruction_time), np.sum(projection_times)*1000/len(projection_times) ));
