#muti file best data gradient descent
obss = ['ener','gradnorm']
fdir = ["./data/ADVUMPS_Heisenberg(2.0, 0.5, 0.5)/", "./data/ADCTMRG_Heisenberg(2.0, 0.5, 0.5)/"]
# %matplotlib notebook
# %matplotlib inline
# from jupyterthemes import jtplot
# jtplot.style(theme='chesterish')

import matplotlib.pyplot as plt
import os  
import numpy as np 
from itertools import groupby
import re
import pandas as pd
import numpy as np

feature = ('D', 'chi','maxiter','tol','time','ener','gradnorm')

def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.log':  
                # L.append(os.path.join(root, file))  
                file_name = file[:]  #去掉.txt
                L.append(file_dir+file_name)  
    return L

def parse(f):
    D = int(re.search('D([0-9]*)_chi', f).group(1))
    chi = int(re.search('chi([0-9]*)_tol', f).group(1))
    maxiter = int(re.search('maxiter([0-9]*).log', f).group(1))
    tol = float(re.search('tol(-?[1-9](?:\.\d+)?[Ee][-+]?\d+)_maxiter', f).group(1))
    return D, chi, maxiter, tol

def keyfunc(f):
    D, chi, maxiter, tol= parse(f)
    return D, chi, maxiter, tol

def get_min_ener(x):
    df = x.sort_values(by = 'ener',ascending=True)
    return df.iloc[0,:]

fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True)
for file_dir in fdir:
    txt_name = file_name(file_dir)
    data = pd.DataFrame(columns=feature)
    files = [list(l) for k, l in groupby(sorted(txt_name, key=keyfunc), keyfunc)]
    print(len(files))

    # markers = ['o', 's', 'D', '*', 'x', '^', '<', 'h']
    # colors = ['C2', 'C1','C0', 'C3', 'C4', 'C5', 'C6', 'C7']
    # for fg, marker, color in zip(files, markers, colors):
    xlist = []
    ylist = []
    for fg in files:
        for f in fg:
            D, chi, maxiter, tol = parse(f)
            time, epoch, ener, gradnorm = np.loadtxt(f, unpack=True)
            if type(epoch) == np.ndarray:
                for i in range(len(epoch)):
                    datarow = pd.DataFrame([[D, chi, maxiter, tol, time[i], ener[i], gradnorm[i]]],columns=feature)
                    data = data.append(datarow,ignore_index=True)
            else:
                datarow = pd.DataFrame([[D, chi, maxiter, tol, time, ener, gradnorm]],columns=feature)
                data = data.append(datarow,ignore_index=True)
                
    for i in range(2):
        obs = obss[i]
        axis=ax[i]
        datadescend = data.loc[data['D']==2].loc[data['chi']==20]
        axis.plot(list(range(datadescend.shape[0]))[0:datadescend.shape[0]], datadescend[obs][0:datadescend.shape[0]],\
                 '-', \
                 label=file_dir)
        axis.set_xlabel('iteration')
        if obs == 'ener':
            axis.set_ylabel('energy')
            axis.legend(loc='upper right')
        elif obs == 'gradnorm':
            axis.set_ylabel('gradnorm')
            axis.set_yscale('log')
            plt.rcParams['axes.unicode_minus'] = False
        else:
            print ('which obs ?')
            sys.exit(1)

    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(8, 12)
fig.savefig("./plot/Heisenberg(2.0, 0.5, 0.5).png", dpi=200)
plt.show()