clear all
close all
% addpath([fileparts(pwd),'/Data']); %add Data directory to path;
addpath('crystal_symmetry_ops');
addpath('octonion_functions/');
addpath('rotation_conversions/');

pgnum = 30; %cubic symmetry
genplot = true;

symnames = load('PGnames.mat'); %need to add crystal_symmetry_ops to path in order for this to work
symops = load('PGsymops.mat');
all_sym = symops.Q{30}; % the quaternions of 24 symmetry operations

data = importdata('all_orientations.txt');
[n_data0, ~] = size(data);
n_data = n_data0/3;

for i= 1:n_data
    init_val =  3*i- 2;
    fin_val = init_val + 2;
    O_1 = data(init_val:fin_val, 1:3);
    O_2 = data(init_val:fin_val, 4:6);
    oct = GBmat2oct(O_1,O_2);
    all_oct(i, :) = oct;
end

pgnum = 30; %cubic symmetry
genplot = false;
oldpd = []; oldoct = []; 
printbool = true; fname = 'little_test.txt';
tic 
pdtest = GBpd(oldpd,oldoct,real(all_oct),pgnum,printbool,fname);
toc
