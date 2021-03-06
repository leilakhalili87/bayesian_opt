# %% [markdown]
# # Noise-free optimization with Expected Improvement

# %%
import numpy as np
import tensorflow as tf
import trieste

import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary, positive, set_trainable
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter


np.random.seed(1)
tf.random.set_seed(1)

# %%
tf.config.run_functions_eagerly(True)
# define database '297' or '388'
N = 297

# define numbe rof training dataset
n_train = 200

n_iter = 1

n_max = 1
# Read the data
if N == 297:
    data_E = '../../../data/297_energy.txt'
    data_pd = '../../../data/297_octonion_pd.txt'
    data_axes = '../../../data/sigma3_data.txt'
    if n_train > N:
        print('The numbe rof trainign datset should be smaller than 297')
else:
    data_E = '../../../data/energy_olms.txt'
    data_pd = '../../../data/pd_olms.txt'
    if n_train > N:
        print('The numbe rof trainign datset should be smaller than 388')
def _octonion_dist(X, X2):
    X = tf.reshape(X, [-1,1])
    pd = np.loadtxt(data_pd)
    X2 = tf.reshape(X2, [-1,1])
    dist0 = np.zeros((len(X), len(X2)))
    dist = tf.Variable(dist0) # Use variable 
    for i in range(len(X)):
        init_val = int(X[i].numpy())
        for j in range(len(X2)):
            fin_val = int(X2[j].numpy())
            dist0[i,j] = pd[init_val, fin_val]
    dist.assign(dist0)
    return dist

def _octonion_dist_single(X):
    X = tf.reshape(X, [-1,1])
    pd = np.loadtxt(data_pd)
    dist0 = np.zeros((len(X)))
    dist = tf.Variable(dist0) # Use variable 
    for i in range(len(X)):
        init_val = int(X[i].numpy())

        dist0[i] = pd[init_val, init_val]
    dist.assign(dist0)
    return dist

class GBKernel(gpflow.kernels.Kernel):
    def __init__(self, variance=1, lengthscales=1, **kwargs):
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)
    
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        a = self.variance * tf.exp(-_octonion_dist(X, X2)/self.lengthscales)
        tf.debugging.check_numerics(self.lengthscales, 'hi')
        return a

    def K_diag(self, X):
        a = self.variance * tf.exp(-_octonion_dist_single(X)/self.lengthscales)
        tf.debugging.check_numerics(self.lengthscales, 'hi')
        return a


# %%
#input data
N = 297
id_gb = np.linspace(0,N-1,N)
id_gb = tf.convert_to_tensor(id_gb, dtype=tf.float64)


# %% [markdown]
# ## Describe the problem
# In this example, we look to find the minimum value of the two-dimensional Branin function over the hypercube $[0, 1]^2$. We can represent the search space using a `Box`, and plot contours of the Branin over this space.

# %%

tf.config.run_functions_eagerly(True)
def gb_func(id_gb):
    """
    x is the id of gbs
    output is the energy
    """
    y0 = -np.loadtxt('../../../data/E_cleav.txt')
#     y0 = -np.loadtxt('../../../data/energy_olms.txt')
    mean_y = np.mean(y0)
    std_y = np.sqrt(np.var(y0))
    y = (y0 - mean_y) / std_y
    id_gb = id_gb.numpy()
    Y = y[id_gb.astype(int)]
    return Y



# %%

input_data =tf.convert_to_tensor(id_gb)
inp = tf.reshape(input_data, [-1,1])
search_space = trieste.space.DiscreteSearchSpace(inp)


# %% [markdown]
# ## Sample the observer over the search space
#
# Sometimes we don't have direct access to the objective function. We only have an observer that indirectly observes it. In _Trieste_, the observer outputs a number of datasets, each of which must be labelled so the optimization process knows which is which. In our case, we only have one dataset, the objective. We'll use _Trieste_'s default label for single-model setups, `OBJECTIVE`. We can convert a function with `branin`'s signature to a single-output observer using `mk_observer`.
#
# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We sample five points from the search space and evaluate them on the observer.

# %%
from trieste.acquisition.rule import OBJECTIVE

observer = trieste.utils.objectives.mk_observer(gb_func, OBJECTIVE)

num_initial_points = 2
initial_query_points = np.array([18, 209])
# initial_query_points = search_space.sample(num_initial_points)
initial_query_points = tf.convert_to_tensor(initial_query_points, dtype=tf.float64)
initial_query_points = tf.reshape(initial_query_points, [-1,1])

initial_data = observer(initial_query_points)


# %%
initial_query_points.shape
# initial_query_points0

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use Gaussian process regression for this, provided by GPflow. The model will need to be trained on each step as more points are evaluated, so we'll package it with GPflow's Scipy optimizer.
#
# Just like the data output by the observer, the optimization process assumes multiple models, so we'll need to label the model in the same way.

# %%
import gpflow

def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = GBKernel()
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
#     print_summary(gpr)
    gpflow.set_trainable(gpr.kernel.lengthscales, True)
    gpflow.set_trainable(gpr.kernel.variance, True)
#     gpflow.set_trainable(gpr.likelihood, True)

    return {OBJECTIVE: {
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    }}

model = build_model(initial_data[OBJECTIVE])

# %% [markdown]
# ## Run the optimization loop
#
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
#
# The optimizer uses an acquisition rule to choose where in the search space to try on each optimization step. We'll use the default acquisition rule, which is Efficient Global Optimization with Expected Improvement.
#
# We'll run the optimizer for fifteen steps.
#
# The optimization loop catches errors so as not to lose progress, which means the optimization loop might not complete and the data from the last step may not exist. Here we'll handle this crudely by asking for the data regardless, using `.try_get_final_datasets()`, which will re-raise the error if one did occur. For a review of how to handle errors systematically, there is a [dedicated tutorial](recovering_from_errors.ipynb). Finally, like the observer, the optimizer outputs labelled datasets, so we'll get the (only) dataset here by indexing with tag `OBJECTIVE`.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
# print_summary(model)
result = bo.optimize(60, initial_data, model)
dataset = result.try_get_final_datasets()[OBJECTIVE]

# %% [markdown]
# ## Explore the results
#
# We can now get the best point found by the optimizer. Note this isn't necessarily the point that was last evaluated.

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
y0 = -np.loadtxt('../../../data/E_cleav.txt')
mean_y = np.mean(y0)
std_y = np.sqrt(np.var(y0))

predicted = (observations * std_y + mean_y)
arg_min_idx = tf.squeeze(tf.argmin(predicted, axis=0))

print(f"grain boundary id: {int(query_points[arg_min_idx, :]) + 1}")
print(f"Predicted value: {-predicted[arg_min_idx, :]}")
print(f"Optimization step: {arg_min_idx}")



# %%
import matplotlib.pyplot as plt
from util.plotting import plot_regret

fig, ax = plt.subplots(figsize=(10, 5))
plot_regret(predicted, ax, num_init=num_initial_points, idx_best=arg_min_idx)
plt.gca().invert_yaxis()
plt.savefig('Step_Eng.jpg')


# %%
# id_gb = np.linspace(0,N-1,N)
# plt.scatter(id_gb, y0)

# %%

ls_list = [
    step.models[OBJECTIVE].model.kernel.lengthscales.numpy()  # type: ignore
    for step in result.history + [result.final_result.unwrap()]
]

var_list = [
    step.models[OBJECTIVE].model.kernel.variance.numpy()  # type: ignore
    for step in result.history + [result.final_result.unwrap()]
]
ls = np.array(ls_list)
var = np.array(var_list)

plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
fig.suptitle('Horizontally stacked subplots')
ax1.plot(ls, 'C3')
ax1.set_xlabel('step')
ax1.set_ylabel('lengthscale')
ax2.plot(var,'C3')
ax2.set_xlabel('step')
ax2.set_ylabel('variance')
plt.subplots_adjust(wspace=0.4)
plt.savefig('var_leng.jpg')
gpflow.utilities.print_summary(result.try_get_final_models()[OBJECTIVE].model)


# %%
my_model = result.try_get_final_models()[OBJECTIVE].model

# %%
id_gb0 = tf.reshape(id_gb, [-1,1])
Fmean, Fvar = my_model.predict_y(id_gb0)
# Fmean = -tf.multiply(Fmean, std_y) - mean_y

# id_gb0 = tf.reshape(id_gb, [-1,1])
Fmean_q, Fvar_q = my_model.predict_y(query_points)

# %%
plt.figure(figsize=(8, 4))
# plt.plot(id_gb, -y0, "kx", mew=2)
min_x = np.linspace(-10, 300, 100)
min_y = min_x*0 + np.min((y0-mean_y)/std_y)
plt.plot(query_points, Fmean_q[:, 0], "kx", lw=2, label="query_points")
plt.fill_between(
    id_gb,
    Fmean[:, 0] - 1.96 * np.sqrt(Fvar[:, 0]),
    Fmean[:, 0] + 1.96 * np.sqrt(Fvar[:, 0]),
    color="C3",
    alpha=0.2, label='Confidence Interval'
)
plt.plot(min_x, min_y, ':', color='r', label="max. Cleavage Energy")
plt.xlim([-5,300])
plt.xlabel('GB ID')
plt.ylabel('Normalized Cleavage Energy')
plt.gca().invert_yaxis()
plt.legend(loc='upper left')
plt.savefig('uncert_60.jpg')

# %%
import pickle as pkl
axes = np.loadtxt(data_axes)[:,0:3].astype(int)
output ={}
output['query_points_id'] = query_points.reshape(-1).astype(int)
output['query_points_hkl'] = axes[query_points.reshape(-1).astype(int)]
output['Predicted_E'] = -predicted.reshape(-1).astype(int)
output['all_gb_id'] = id_gb.numpy().astype(int)
output['all_gb_hkl'] = axes[id_gb.numpy().reshape(-1).astype(int)]
output['Cleavage_E_all_gb'] = -y0
bo = open('BO_output.pkl', 'wb')
pkl.dump(output, bo)
bo.close()


# %%
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.size'] = 20
x = u1_hkl[:,0]
y = u1_hkl[:,1]
z = u1_hkl[:,2]



xq = q1_hkl[:,0]
yq = q1_hkl[:,1]
zq = q1_hkl[:,2]

max_x= q1_hkl[arg_min_idx,0]
max_y= q1_hkl[arg_min_idx,1]
max_z= q1_hkl[arg_min_idx,2]

ph1 = np.arctan2(y,x)
th1 = np.arccos(z)

ph1_max = np.arctan2(max_y,max_x)
th1_max = np.arccos(max_z)

ph1q = np.arctan2(yq,xq)
th1q = np.arccos(zq)

X = np.sqrt(2*(1 - np.abs(np.cos(th1))))*np.cos(ph1)
Y = np.sqrt(2*(1 - np.abs(np.cos(th1))))*np.sin(ph1)

Xq = np.sqrt(2*(1 - np.abs(np.cos(th1q))))*np.cos(ph1q)
Yq = np.sqrt(2*(1 - np.abs(np.cos(th1q))))*np.sin(ph1q)

X_max = np.sqrt(2*(1 - np.abs(np.cos(th1_max))))*np.cos(ph1_max)
Y_max = np.sqrt(2*(1 - np.abs(np.cos(th1_max))))*np.sin(ph1_max)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.box(False)
plt.xticks([])
plt.yticks([])
# plt.scatter(X, Y, alpha=.3)
p=plt.scatter(Xq, Yq, c=Predicted_E, vmin=1520, vmax=1850, cmap='jet', s=100, alpha=2)
# p = plt.scatter(X, Y, c=Cleavage_E_all_gb, cmap='jet')
plt.scatter(X, Y, alpha=.1)
plt.scatter(X_max, Y_max, c=Predicted_E[arg_min_idx], marker='*', s=200)
plt.xlim([0-0.1, np.sqrt(2)*1.1])
plt.ylim([0-0.1, np.sqrt(2)*.7])
ax.set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
cbar = fig.colorbar(p, cax=cax, ticks=[1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800,1840])
cbar.ax.set_ylabel('mJ/$m^2$', loc='top')
# plt.xlim([-1,15])
# plt.ylim([-1,15])
plt.savefig('fz.jpg')
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
plt.scatter(Xq, Yq, marker='x', s=150, color='r', alpha=1)
plt.scatter(X, Y, c=Cleavage_E_all_gb, s=650, cmap='viridis', edgecolor='none', alpha=.1)

# %%
predicted_E

# %%
Predicted_E

# %%
