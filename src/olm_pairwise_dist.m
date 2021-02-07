clear all
clc
%% 1. LOAD OCTONION UTILITY FUNCTIONS 

% addpath([fileparts(pwd),'/Data']); %add Data directory to path;
addpath('crystal_symmetry_ops');
addpath('octonion_functions/');
addpath('rotation_conversions/');

pgnum = 30; %cubic symmetry
genplot = true;

symnames = load('PGnames.mat'); %need to add crystal_symmetry_ops to path in order for this to work
symops = load('PGsymops.mat');
all_sym = symops.Q{30}; % the quaternions of 24 symmetry operations

% import Olmsted dataset
% olmx = importdata('dat_leila.txt');
olmimp = importdata('olmsted_xtal_info_numeric.csv');
olmx = olmimp.data; ngb = length(olmx);
feature_names = olmimp.textdata;
[ngb, ~] = size(olmx);


% extract indices of orientation matrices as N_GB x 9 matrices
O1mat = olmx(:,[5:7 11:13 17:19]);
O2mat = olmx(:,[8:10 14:16 20:22]);

octlist = zeros(ngb,8);

for i = 1:ngb
    % O1 and O2 should have hkl directions along columns (x,y,z)
    O1 = reshape(O1mat(i,:),[3,3]);
    O = reshape(O2mat(i,:),[3,3]);
    
    aa_z = [0 1 0 pi/2];
    om_z = ax2om(aa_z); %rotation matrix, BP x --> z 
    
    OA = (om_z*O1')'; %rotate row-wise, transpose to column form
    OB = (om_z*O')';
    oct = GBmat2oct(OA,OB);
    
    misorien = qu2om(qmult(OB, qinv(OA)));
    ax_ang = qu2ax(misorien);
    ax = ax_ang(:,1:3)/norm(ax_ang(:,1:3));
    angle = ax_ang(:,4);
    [Theta,R] = stereo(ax);
    x(i) = R*sin(angle)*cos(Theta);
    y(i) = R*sin(angle)*sin(Theta);
    z(i) = R*cos(angle);
    
%     all_oct(i, :) = oct;
%     if i >1
%         oct_pair(i, :) = [all_oct(1,:), oct]; 
%     end
end
test = importdata('../data/olm_octonion_list.txt',' ',1); %list of GB octonions with number of octonions as first line in file
data0 = test.data;

% as a simple example, we will fold the Olmsted dataset in half to give a 194x16 matrix of GB pairs
data = zeros(388,16);
for i=1:388
    data(i,1:8) = data0(1,:);
end
data(:,9:16) = data0(1:388,:);

pgnum = 30; %cubic symmetry
genplot = false;

%this takes 48.2 seconds on a single core (which is quite slow). A much faster implementation of a
%similar routine can be found in the EMSOFT programs EMGBO / EMGBOdm

tic 
[omega_test, oct_test, zeta_test] = GBdist(data, pgnum, genplot);
toc 

figure
scatter3(x, y, z, 388,  omega_test, 'filled')
xlabel('x')
ylabel('y')
zlabel('z')
colorbar('Location', 'EastOutside', 'YTickLabel',...
    {'0', '.25', '.5', '.75','1'})