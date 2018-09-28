import cPickle as pickle
#
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold
#
import pp
import os
import time
import sys
import copy
###################################################################
#
#  SIMPLE PROGRESS BAR
#
###################################################################
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
###################################################################
#
#        KERNEL DENSITY ESTIMATION (KDE) TO FIND MOST PROBABLE VALUES
#               (cross-validated bandwidth)
#
###################################################################
def best_KDE_estimator_multivar(data, kern = 'gaussian', cv_num = 10,r_tolerance=1e-8, ncpus = 8):
    grid = GridSearchCV(KernelDensity(kernel = kern,rtol = r_tolerance), {'bandwidth': np.logspace(-3, 1, 25)}, cv= cv_num,n_jobs = ncpus)
    grid.fit(data)
    return grid.best_estimator_

def best_estimators_multivar(coords, kernel, CV_num, R_tolerance, Ncpus = 8):
    estimators={}
    bandwidth = np.zeros(coords.shape[2])
    for NumAtom in range(0,coords.shape[2]):
        dat = coords[:,:,NumAtom]
        estimators[NumAtom]=best_KDE_estimator_multivar(dat,kern = kernel,cv_num = CV_num,r_tolerance=R_tolerance,ncpus=Ncpus)
        bandwidth[NumAtom] = estimators[NumAtom].bandwidth
        progress(NumAtom+1,bandwidth.shape[0])
    return estimators, bandwidth

def estimate_max_probability(coordinates,estimators):
    import numpy as np
    output = np.zeros((coordinates.shape[2],3))
    for NumAtom in range(0,coordinates.shape[2]):
        dat = coordinates[:,:,NumAtom]
        estimator = estimators[NumAtom]
        vals = np.exp(estimator.score_samples(dat))
        output[NumAtom,:] = dat[np.where(vals==np.max(vals))[0][0]]
        #progress(NumAtom+1,coordinates.shape[2])
    return output

def prob(xyz_, *args):
    estimator, = args
    xyz_=xyz_.reshape(-1,1).T
    return -(estimator.score_samples(xyz_))

def PSO_most_probable(estimators_,lb, ub, swarmsize=100,maxiter=500, minstep = 1e-12,minfunc=1e-16):
    from pyswarm import pso
    import numpy as np
    keys = estimators_.keys()
    output = np.zeros((len(keys),3))
    for (i,key) in enumerate(estimators_.keys()):
        estimator = estimators_[key]
        xopt, fopt = pso(prob, lb, ub, args=(estimator,),swarmsize=swarmsize,maxiter=maxiter, minstep = minstep,minfunc=minfunc)
        output[i,:] = xopt
    return output

def MostProbablePositions_PSO(estimators,lb, ub, SwarmSize=100,MaxIter=500, min_step = 1e-12,est_tolerance = 1e-16, ncpus = 8):
    print('Estimating most probable coordinates')
    Nstart = 0
    Nstop = len(estimators.keys())
    step = (Nstop - Nstart) / ncpus + 1
    job_server = pp.Server()
    job_server.set_ncpus(ncpus)
    jobs = []
    start_time = time.time()
    for index in xrange(ncpus):
        starti = Nstart+index*step
        endi = min(Nstart+(index+1)*step, Nstop)
        estimat_dict_submit = {}
        for j in range(starti,endi):
            estimat_dict_submit[j-starti] = copy.deepcopy(estimators[j])
        jobs.append(job_server.submit(PSO_most_probable, (estimat_dict_submit,lb, ub, SwarmSize,MaxIter, min_step,est_tolerance),(prob,)))
    job_server.wait()
    print('-------------------------------------------------------------------------')
    print('Time elapsed %d sec' % (time.time() - start_time))
    print('-------------------------------------------------------------------------')
    job_server.print_stats()
    for i in range (0,len(jobs)):
        if i==0:
            output = jobs[i]()
        else:
            output = np.append(output,jobs[i](), axis=0)
    job_server.destroy()    
    return output

def MostProbablePositions_multivar(coordinates, kernel = 'gaussian', r_tolerance = 1e-8 ,ncpus= 8, cv_num = 20,job_name=''):
    fname2search = job_name + '_estimators_' + kernel+'_tolerance_'+str(r_tolerance)+'.pkl'
    if os.path.isfile(fname2search):
        print('Found calculated estimators for job: %s with kernel: %s and tolerance %f' % (job_name,kernel,r_tolerance))
        with open(fname2search, 'rb') as handle:
            estimators = pickle.load(handle)
    else:
        print('No pre-calculated data is found')
        print('Searching for the best-fit bandwidth')
        start_time = time.time()
        estimators, bandwidth = best_estimators_multivar(coordinates, kernel, cv_num, r_tolerance, ncpus)
        print('------------------------------------------------------------------------------------')
        print('Time elapsed %d sec' % (time.time() - start_time))
        print('------------------------------------------------------------------------------------')
        try:
            np.savetxt(job_name+'_bandwidthes_kernel_' + kernel+'_tolerance_'+str(r_tolerance)+'.dat',bandwidth,fmt='%22.18f')
        except:
            pass
        with open(job_name + '_estimators_' + kernel+'_tolerance_'+str(r_tolerance)+'.pkl', 'wb') as handle:
            pickle.dump(estimators, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output = np.zeros((coordinates.shape[2],3))
    print('Estimating most probable coordinates')
    Nstart = 0
    Nstop = len(estimators.keys())
    step = (Nstop - Nstart) / ncpus + 1
    job_server = pp.Server()
    job_server.set_ncpus(ncpus)
    jobs = []
    start_time = time.time()
    for index in xrange(ncpus):
        starti = Nstart+index*step
        endi = min(Nstart+(index+1)*step, Nstop)
        XYZ = coordinates[:,:,starti:endi]
        estimat_dict_submit = {}
        for j in range(starti,endi):
            estimat_dict_submit[j-starti] = copy.deepcopy(estimators[j])
        jobs.append(job_server.submit(estimate_max_probability, (XYZ,estimat_dict_submit),))
    job_server.wait()
    print('-------------------------------------------------------------------------')
    print('Time elapsed %d sec' % (time.time() - start_time))
    print('-------------------------------------------------------------------------')
    job_server.print_stats()
    for i in range (0,len(jobs)):
        if i==0:
            output = jobs[i]()
        else:
            output = np.append(output,jobs[i](), axis=0)
    job_server.destroy()    
    return output


def MostProbablePositions_multivar_PSO(coordinates, lb, ub, kernel = 'gaussian', r_tolerance = 1e-8 ,ncpus= 8, cv_num = 20,job_name='', SwarmSize=100,MaxIter=500, min_step = 1e-12,est_tolerance = 1e-16):
    """
    1. Identifies best bandwidth for the given kernel using cross-validation
    2. Performs PSO search of the most probable position confined by lb, ub arrays
    """
    fname2search = job_name + '_estimators_' + kernel+'_tolerance_'+str(r_tolerance)+'.pkl'
    if os.path.isfile(fname2search):
        print('Found calculated estimators for job: %s with kernel: %s and tolerance %f' % (job_name,kernel,r_tolerance))
        with open(fname2search, 'rb') as handle:
            estimators = pickle.load(handle)
    else:
        print('No pre-calculated data is found')
        print('Searching for the best-fit bandwidth')
        start_time = time.time()
        estimators, bandwidth = best_estimators_multivar(coordinates, kernel, cv_num, r_tolerance, ncpus)
        print('------------------------------------------------------------------------------------')
        print('Time elapsed %d sec' % (time.time() - start_time))
        print('------------------------------------------------------------------------------------')
        try:
            np.savetxt(job_name+'_bandwidthes_kernel_' + kernel+'_tolerance_'+str(r_tolerance)+'.dat',bandwidth,fmt='%22.18f')
        except:
            pass
        with open(job_name + '_estimators_' + kernel+'_tolerance_'+str(r_tolerance)+'.pkl', 'wb') as handle:
            pickle.dump(estimators, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output = np.zeros((coordinates.shape[2],3))
    output = MostProbablePositions_PSO(estimators,lb, ub, SwarmSize,MaxIter, min_step,est_tolerance)
    return output, estimators
