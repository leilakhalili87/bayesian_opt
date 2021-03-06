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

% olmx= importdata('orientations0.txt'); % read the orientations
% O1 = olmx(:,1:3);
% O = olmx(:,4:6);
olmimp = importdata('olmsted_xtal_info_numeric.csv');
olmx = olmimp.data; ngb = length(olmx);
feature_names = olmimp.textdata;
[ngb, ~] = size(olmx);


% extract indices of orientation matrices as N_GB x 9 matrices
O1mat = olmx(:,[5:7 11:13 17:19]);
O2mat = olmx(:,[8:10 14:16 20:22]);
O1 = reshape(O1mat(3,:),[3,3]);
O = reshape(O2mat(3,:),[3,3]);
aa_z = [0 1  0 pi/2];
om_z = ax2om(aa_z); %rotation matrix, BP y --> z 
OA1 = (om_z*O1')'; %rotate row-wise, transpose to column form
OB1 = (om_z*O')';

oct1 = GBmat2oct(OA1,OB1); % covert the orietnation into quaternions
OA = oct1(1:4);
OB = oct1(5:8);

% ==========================================
% calculate the distance
for i =1:24
    q_rot0 = all_sym(i,:);
    ax_ang = qu2ax(q_rot0);
    ax_ang(4) = ax_ang(4) + .01;
    q_rot = ax2qu(ax_ang);
    A = qmult(q_rot, OA);
    B = qmult(q_rot, OB);
    oct2 = [A,B];
    ans0(i) = GBdist([oct1, oct2], pgnum, genplot);
end
ans0


