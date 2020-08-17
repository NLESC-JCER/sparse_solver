#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import re

from os import listdir
from os.path import isfile, join

import collections


# In[ ]:





# In[ ]:





# In[2]:


def parselogfile(filename):
    found_profile=False
    data=collections.defaultdict(dict)
    pattern="\[  ([^1-9]+): +(\d+\.\d+) s\]"
    with open(filename,"r") as f:
        for line in f:
            if "iter:" in line and "error " in line:
                parts=line.split()
                if(len(parts))!=4:
                    print("Wrong number of entries")
                data[parts[0]]["iters"]=int(parts[1].split(":")[1])
            if "[Profile:       " in line:
                found_profile=True
            if found_profile:
                match = re.match(pattern, line)
                if match:
                    for stage in ["setup","solve"]:
                        if stage in match.group(1):
                            name=match.group(1).replace(stage+"_","")
                            data[name][stage]=float(match.group(2))       
    return data


# In[3]:


def parsefolder(foldername,threads):
    logfiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
    print("Found {} files in folder '{}'".format(len(logfiles),foldername))
    results={}
    for f in logfiles:
        data=parselogfile(foldername+'/'+f)
        
        seq= int(f.split("_")[0][-1])
        number=int(f.split("_")[1].split(".")[0])
        for name,result in data.items():
            name="{}_{}".format(threads,name)
            if seq not in results:
                results[seq]={}
            if name not in results[seq]:
                results[seq][name]={}
            if number not in results[seq][name]:
                results[seq][name][number]={}
            results[seq][name][number]=result
    return results

    


# In[4]:


def MergeResults(listofresults):
    results=listofresults[0]
    for r in listofresults[0:]:
        for seq in r.keys():
            results[seq].update(r[seq])
    return results

results=[]
for f,t in zip(["cuda_logs","1_threads","2_threads","4_threads"],[0,1,2,4]):
    results.append(parsefolder(f,t))
    
result=MergeResults(results)
#print(result)



def getNames(results):
    temp_names=[]
    for seq in results.keys():
        for name in results[seq].keys():
            temp_names.append(name)
    return list(set(temp_names)) 


   

def converttoArrays(results):
    namelist=getNames(results)
    no_of_timesteps=len((results[1][namelist[0]]))
    names_index={k: v for v, k in enumerate(namelist)}
    stages_index={'iters':0,'setup':1,'solve':2}
    seqs=len(results)
    print("Creating array of dimension {}x3x{}x{}".format(seqs,len(namelist),no_of_timesteps))
    data=np.zeros((seqs,3,len(namelist),no_of_timesteps)) #four dimensional first dimension seqs,second dim [iters,setup,solve], third[name], fourth timestep
    for seq, names in results.items():
        for name, timesteps in names.items():
            for i,(t,stages) in enumerate(timesteps.items()):
                for stage, value in stages.items():
                    data[seq-1,stages_index[stage],names_index[name],i]=value
    return data,namelist,['iters','setup','solve']

data,names,stages=converttoArrays(result)


# In[26]:


print(names)
totaltimings=data[:,1,:,:]+data[:,2,:,:]
averagetimings=np.average(totaltimings,axis=2)
std_timings=np.std(totaltimings,axis=2)
iters=data[:,0,:,:]
#print(iters)
averageiters=np.average(iters,axis=2)
#print(iters)
std_iters=np.std(iters,axis=2)
#print(averageiters)
#print(std_iters)



def selectarray(pattern,names,array,error):
    small_names=[]
    logic_array=[]
    for name in names:
        match = re.match(pattern, name)
        if match:
            small_names.append(name)
            logic_array.append(True)
        else:
            logic_array.append(False)
    
    return small_names,array[:,logic_array],error[:,logic_array]

def printbarchart(names,array,error):

    bottom=np.zeros(len(names))
    for seq in range(4):
        plt.bar(names,array[seq],yerr=error[seq], label=seq+1,bottom=bottom)
        bottom+=array[seq]
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()  

name_4,iters_4,std_iters_4=selectarray("[04]?_",names,averageiters,std_iters)
printbarchart(name_4,iters_4,std_iters_4)


name_4,time_4,std_time_4=selectarray("[04]?_",names,averagetimings,std_timings)
printbarchart(name_4,time_4,std_time_4)


# In[33]:


name_scaling,time_scaling,std_time_scaling=selectarray("[124]?_amgcl_bicgstab_ilut$",names,averagetimings,std_timings)
printbarchart(name_scaling,time_scaling,std_time_scaling)


# In[32]:


name_scaling,time_scaling,std_time_scaling=selectarray("[124]?_eigen_bicgstab$",names,averagetimings,std_timings)
printbarchart(name_scaling,time_scaling,std_time_scaling)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




