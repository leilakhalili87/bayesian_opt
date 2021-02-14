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

data = importdata('../data/sigma3_data.txt');
energy = data(:,4);
cleavage = data(:,5);
miller = data(:,1:3);
[n_data, ~] = size(data);

T = [[-1, 2, 2];[2, -1, 2];[2, 2, -1]]/3;

z = [0,0,1];
for i=1:n_data
    hkl = miller(i,:);

    n_hkl = hkl/norm(hkl);
    rot_ax = cross(n_hkl, z);
    rot_ax = normr(rot_ax);
    cos_ang = dot(n_hkl, z);
    ang = acos(cos_ang);
    ax_ang = [rot_ax, ang];
    O_1 = vrrotvec2mat(ax_ang);
    O_2 = T*O_1;
    all_oct(i,:) = GBmat2oct(O_1,O_2);
end

% example
% db_1 = [[2,1,1];[2,-1,1];[2,0,-2]];
% db_2 = [[2,-1,-1];[2,1,-1];[2,0,2]];

for i=1:(n_data-1)
    Data(i,1:8) = all_oct(1,:);
end
Data(:,9:16) = all_oct(2:n_data,:);

pgnum = 30; %cubic symmetry
genplot = false;
oldpd = []; oldoct = []; 
printbool = true; fname = 'little_test.txt';
tic 
pdtest = GBpd(oldpd,oldoct,all_oct,pgnum,printbool,fname);
toc



