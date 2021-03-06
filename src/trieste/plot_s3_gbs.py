#!/bin/zsh

import numpy as np

def rot_bpls(bpns, Rmat):
	bpn0 = bpns.transpose()
	return (np.dot(Rmat, bpn0)).transpose()

## Z-axis
zvec = np.array([[1], [1], [1]])
zvec = zvec/np.linalg.norm(zvec)

## X-axis
xvec = np.array([[2], [-1], [-1]])
xvec = xvec/np.linalg.norm(xvec)

## Y-axis
yvec = (np.cross(zvec.transpose(), xvec.transpose())).transpose()

M1 =  np.hstack((xvec, yvec, zvec))

Rmat = np.linalg.inv(M1)

###############################################################
import pickle as pkl
jar = open('BO_output.pkl','rb')
s1 = pkl.load(jar)
jar.close()

query_points_id = s1['query_points_id']
query_points_hkl = s1['query_points_hkl']
Predicted_E = s1['Predicted_E']
all_gb_id = s1['all_gb_id']
all_gb_hkl = s1['all_gb_hkl']
Cleavage_E_all_gb = s1['Cleavage_E_all_gb']

u_hkl = all_gb_hkl/np.tile(np.sqrt(all_gb_hkl[:,0]**2   + all_gb_hkl[:,1]**2 + all_gb_hkl[:,2]**2), (3,1)).transpose()
u1_hkl = rot_bpls(u_hkl, Rmat)

q_hkl = query_points_hkl/np.tile(np.sqrt(query_points_hkl[:,0]**2   + query_points_hkl[:,1]**2 + query_points_hkl[:,2]**2), (3,1)).transpose()
q1_hkl = rot_bpls(q_hkl, Rmat)

import matplotlib.pyplot as plt
x = u1_hkl[:,0]
y = u1_hkl[:,1]
z = u1_hkl[:,2]

xq = q1_hkl[:,0]
yq = q1_hkl[:,1]
zq = q1_hkl[:,2]

ph1 = np.arctan2(y,x)
th1 = np.arccos(z)

ph1q = np.arctan2(yq,xq)
th1q = np.arccos(zq)

X = np.sqrt(2*(1 - np.abs(np.cos(th1))))*np.cos(ph1)
Y = np.sqrt(2*(1 - np.abs(np.cos(th1))))*np.sin(ph1)

Xq = np.sqrt(2*(1 - np.abs(np.cos(th1q))))*np.cos(ph1q)
Yq = np.sqrt(2*(1 - np.abs(np.cos(th1q))))*np.sin(ph1q)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

# plt.scatter(X, Y, alpha=.3)
plt.scatter(Xq, Yq, marker='x', s=150, color='r', alpha=1)
plt.scatter(X, Y, c=Cleavage_E_all_gb)
plt.xlim([0-0.1, np.sqrt(2)*1.1])
plt.ylim([0-0.1, np.sqrt(2)*1.1])
ax.set_aspect('equal', adjustable='box')
plt.show()
###############################################################






