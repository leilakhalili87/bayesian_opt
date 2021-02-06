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
O1 = olmx(:,1:3);
O = olmx(:,4:6);

aa_z = [1 0 0 pi/2];
om_z = ax2om(aa_z); %rotation matrix, BP y --> z 
OA1 = (om_z*O1')'; %rotate row-wise, transpose to column form
OB1 = (om_z*O')';
oct1 = GBmat2oct(OA1,OB1); % covert the orietnation into quaternions


olmx= importdata('orientations_1.txt'); % read the orientations
O1 = olmx(:,1:3);
O = olmx(:,4:6);

aa_z = [1 0 0 pi/2];
om_z = ax2om(aa_z); %rotation matrix, BP y --> z 
OA1 = (om_z*O1')'; %rotate row-wise, transpose to column form
OB1 = (om_z*O')';
oct2 = GBmat2oct(OA1,OB1); % covert the orietnation into quaternions



 ans = GBdist([oct1, oct2], pgnum, genplot)
% ==========================================
% calculate the distance
% for i =1:24
%     q_rot = all_sym(1,:);
%     A = qmult(q_rot, OA);
%     B = qmult(q_rot, OB);
%     oct2 = [A,B];
%     ans0(i) = GBdist([oct1, oct2], pgnum, genplot);
% end
% ans0


