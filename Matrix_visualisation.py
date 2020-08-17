#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


# In[4]:


import scipy.io as scp
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [14, 14]


# In[28]:


path1="seq1"
files = [f for f in listdir(path1) if  isfile(join(path1, f) )]
files = [f for f in files if "A_mat" in f]
print(files)


# In[29]:


A=scp.mmread(path1+"/"+files[0])
plt.spy(A,markersize=1)
plt.savefig("spy.png")
plt.show()


# In[6]:



class diagonal:
    
    def __init__(self,x0,y0):
        self.x0=x0
        self.y0=y0
        self.entries=[]
    
    def append(self,value):
        self.entries.append(value)
        
    def getX(self):
        return self.x0+len(self.entries)-1
   
    def getY(self):
        return self.y0+len(self.entries)-1
    
    def size(self):
        return len(self.entries)
        

class diagonalmatrix:
    
    def __init__(self):
        self.diagonals=[]
        
    def AppendToDiagonal(self,i,j,value):
        for diag in self.diagonals:
            if diag.getX()+1==i and diag.getY()+1==j:
                diag.append(value)
                break;
        else:
            diag=diagonal(i,j)
            diag.append(value)
            self.diagonals.append(diag)
        
    
    def fill(self,matrix):
        for i,j,v in zip(A.row, A.col, A.data):
            self.AppendToDiagonal(i,j,v)

    def info(self):
        print("Diags {}:".format(len(self.diagonals)))
        elements=0
        for diag in self.diagonals:
            elements+=diag.size()
        print("Contains {} elements".format(elements))
            
    


# In[7]:


def getMaxBand(matrix):
    maxband=0
    for i,j in zip(matrix.row, matrix.col):
        maxband=max(maxband,abs(i-j))
    return maxband
    
    
def getSparsity(matrix):
    return (matrix.nnz/float(matrix.shape[0]*matrix.shape[1]))


# In[22]:


values1,edges1=np.histogram(A.row,bins=A.shape[0])


# In[28]:


edges1a=0.5*(edges1[:-1]+edges1[1:])
print(values1.shape)
print(edges1a.shape)
values1.sort()
plt.plot(edges1a,values1)


# In[10]:


#print(getMaxBand(A))

#print(getSparsity(A))
dist=[]
for i,j in zip(A.row, A.col):
        dist.append((i-j))
        
values,edges=np.histogram(dist,bins=A.shape[0])
edges=0.5*(edges[:-1]+edges[1:])


# In[11]:


edges=edges[values>0]
values=values[values>0]
print(values.shape)
density=values/values.sum()
plt.xlim(-200,200)
plt.bar(edges,density)
plt.xlabel("i-j")
plt.savefig("hist.png")
plt.show()

sorted_density=-np.sort(-density)
print(np.cumsum(sorted_density)[0:50])
#plt.plot(sorted_density)
plt.plot(np.cumsum(sorted_density),marker="o")
#plt.plot(np.ones(sorted_density.shape))
plt.xlim(0,20)
plt.ylim(0,1)
plt.ylabel("fillfactor of subdiagonal")
plt.ylabel("subdiagonal sorted")
plt.savefig("hist2.png")
plt.show()


# In[12]:


C=scp.mmread(files[2])
plt.spy(C,markersize=1)
plt.show()


# In[13]:


print(A.nnz)
print(A.shape[0]*A.shape[1])
print(A.nnz/float(A.shape[0]*A.shape[1]))


# In[14]:


plt.spy(A,markersize=1)
size=250
offset=114500
plt.xlim(offset, offset+size)
plt.ylim(offset+size,offset)
plt.show()


# In[ ]:





# In[ ]:





# In[15]:


""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


""


# In[ ]:




