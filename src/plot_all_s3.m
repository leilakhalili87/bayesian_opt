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
x = [1, 1, 0];
[n_data, ~] = size(data);

T = [[-1, 2, 2];[2, -1, 2];[2, 2, -1]]/3;

for i=1:n_data
    y = miller(i,:);
%     y = y/gcd(gcd(y(1), y(2)), y(3));
    
    z = cross(x, y);
%     z = z/gcd(gcd(z(1), z(2)), z(3));
    
    O_1 = [x',y', z'];
    O_2 = T*O_1;

    aa_z = [1 0 0 pi/2];
    om_z = ax2om(aa_z); %rotation matrix, BP y --> z 
    
    O_1 = (om_z*O_1')'; %rotate row-wise, transpose to column form
    O_2 = (om_z*O_2')';
    init_val =  3*i- 2;
    fin_val = init_val + 2;
    
    O(init_val:fin_val, 1:3) = O_1;
    O(init_val:fin_val, 4:6) = O_2;
    O_1
    O_2
    
    oct = GBmat2oct(O_1,O_2);
    all_oct(i, :) = oct;
%     u_ax = z/norm(z);
%     Theta = acos(u_ax(1));
%     phi = acos(u_ax(3));
%     [Theta, R] = stereo(u_ax);
%     X(i) = u_ax(1);
%     Y(i) = u_ax(2);
%     Z(i) = u_ax(3);
end

% for i=1:(n_data-1)
%     Data(i,1:8) = all_oct(1,:);
% end
% Data(:,9:16) = all_oct(2:n_data,:);

pgnum = 30; %cubic symmetry
genplot = false;
oldpd = []; oldoct = []; 
printbool = true; fname = 'little_test.txt';
tic 
pdtest = GBpd(oldpd,oldoct,all_oct,pgnum,printbool,fname);
toc



% X = X(2:297);
% Y = Y(2:297);
% Z = Z(2:297);
% 
% [omega_test, oct_test, zeta_test] = GBdist(Data, pgnum, genplot);
% scatter3(X, Y,Z, 296, abs(omega_test), 'filled')
% xlabel('x')
% ylabel('y')
% zlabel('z')
% colorbar('Location', 'EastOutside', 'YTickLabel',...
%     {'0', '.25', '.5', '.75','1', '1.25', '1.5', '1.75', '2'})
% 
