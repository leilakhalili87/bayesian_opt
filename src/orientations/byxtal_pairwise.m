clear all
clc
%% 1. LOAD OCTONION UTILITY FUNCTIONS 

% addpath([fileparts(pwd),'/Data']); %add Data directory to path;
addpath('crystal_symmetry_ops');
addpath('octonion_functions/');
addpath('rotation_conversions/');

pgnum = 30; %cubic symmetry
genplot = false;

symnames = load('PGnames.mat'); %need to add crystal_symmetry_ops to path in order for this to work
symops = load('PGsymops.mat');
all_sym = symops.Q{30}; % the quaternions of 24 symmetry operations

olmx= importdata('all_orientations.txt'); % read the orientations
[size_data, ~] = size(olmx);
size_data = size_data/3;

OL_ref = olmx(1:3,1:3);
OU_ref = olmx(4:6,4:6);

aa_z = [1 0 0 pi/2];
om_z = ax2om(aa_z); %rotation matrix, BP y --> z 

R_L_ref = (om_z*OL_ref')'; %rotate row-wise, transpose to column form
R_U_ref = (om_z*OU_ref')';
oct_ref = GBmat2oct(R_L_ref,R_U_ref); % covert the orietnation into quaternions


for i=2:size_data
    init_val = 3*(i-1) + 1;
    fin_val = init_val + 2;
    OL = olmx(init_val:fin_val, 1:3);
    OU = olmx(init_val:fin_val, 4:6);
    R_L = (om_z*OL')'; %rotate row-wise, transpose to column form
    R_U = (om_z*OU')';
    oct_dum = GBmat2oct(R_L,R_U); % covert the orietnation into quaternions
    all_oct(i-1, :) = oct_dum;
end
oldpd = []; oldoct = [];
fname = 'little_test.txt';printbool = true;
pdtest = GBpd(oldpd,oldoct,all_oct,pgnum,printbool,fname);