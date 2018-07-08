# -*- coding=utf-8 -*-

import subprocess, sys, os, time

NR_THREAD = 12

start = time.time()


cmd = '/home/liuhui/anaconda3/bin/python parallelizer-ffm.py -s {nr_thread} ./pre-ffm.py data/tr_50_FE.csv data/tr_50_FE.ffm'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True) 
cmd = '/home/liuhui/anaconda3/bin/python parallelizer-ffm.py -s {nr_thread} ./pre-ffm.py data/va_50_FE.csv data/va_50_FE.ffm'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)
cmd = '/home/liuhui/anaconda3/bin/python parallelizer-ffm.py -s {nr_thread} ./pre-ffm.py data/ts_FE.csv data/ts_FE.ffm'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)  


print('time used = {0:.0f}'.format(time.time()-start))
