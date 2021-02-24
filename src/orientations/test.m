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

% import Olmsted dataset
olmx = importdata('dat_leila.txt');
[ngb, ~] = size(olmx);


% extract indices of orientation matrices as N_GB x 9 matrices
O1mat = olmx(:,[5:7 11:13 17:19]);
O2mat = olmx(:,[8:10 14:16 20:22]);
i = 3;
O1 = reshape(O1mat(i,:),[3,3]);
O = reshape(O2mat(i,:),[3,3]);

aa_z = [0 1 0 pi/2];
om_z = ax2om(aa_z); %rotation matrix, BP x --> z 

OA = (om_z*O1')'; %rotate row-wise, transpose to column form
OB = (om_z*O')';

aa_z = [1 0  0 pi/2];
om_z = ax2om(aa_z); %rotation matrix, BP y --> z 
OA1 = (om_z*O1')'; %rotate row-wise, transpose to column form
OB1 = (om_z*O')';

oct1 = GBmat2oct(OA1,OB1); % covert the orietnation into quaternions
OA = oct1(1:4);
OB = oct1(5:8);

% ==========================================
% calculate the distance
for i =1:24
    q_rot = all_sym(i,:);
    q_rot = qinv(q_rot);
    A = qmult(q_rot, OA);
    B = qmult(q_rot, OB);
    
    misorien = qu2om(qmult(B, qinv(A)));
    ax_ang = qu2ax(misorien);
    ax = ax_ang(:,1:3);
    angle = ax_ang(:,4);
    [Theta,R] = stereo(ax);
    x(i) = R*sin(angle)*cos(Theta);
    y(i) = R*sin(angle)*sin(Theta);
    z(i) = R*cos(angle);

    oct = [A,B];
    all_oct(i, :) = oct;
end
oldpd = []; oldoct = [];
fname = 'little_test.txt';printbool = true;
pdtest = GBpd(oldpd,oldoct,all_oct,pgnum,printbool,fname);

