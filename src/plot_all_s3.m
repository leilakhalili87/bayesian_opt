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
    z = miller(i,:);
    y = cross(z, x);
    if norm(y)==0
        i
    end
    init_val =  3*i- 2;
    fin_val = init_val + 2;
    O_1 = [x',y', z'];
    O_2 = T*O_1;
    O(init_val:fin_val, 1:3) = O_1;
    O(init_val:fin_val, 4:6) = O_2;
    oct = GBmat2oct(O_1,O_2);
    all_oct(i, :) = oct;
    
    ax = z/norm(z);
    [Theta,R] = stereo(ax);
    X(i) = Theta;
    Y(i) = R;
end

for i=1:(n_data-1)
    Data(i,1:8) = all_oct(1,:);
end
Data(:,9:16) = all_oct(2:n_data,:);

pgnum = 30; %cubic symmetry
genplot = false;


[omega_test, oct_test, zeta_test] = GBdist(Data, pgnum, genplot);
scatter(X, Y,296, abs(omega_test), 'filled')
xlabel('x')
ylabel('y')
zlabel('z')
colorbar('Location', 'EastOutside', 'YTickLabel',...
    {'0', '.25', '.5', '.75','1', '1.25', '1.5', '1.75', '2'})

