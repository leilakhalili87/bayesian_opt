clear all
close all
pgnum = 30; %cubic symmetry
genplot = false;
% addpath([fileparts(pwd),'/Data']); %add Data directory to path;
addpath('crystal_symmetry_ops');
addpath('octonion_functions/');
addpath('rotation_conversions/');

d = [0.958208598778626  -0.115248086681799  -0.261184080498726   0.018358535789993];
c = [0.693454162735472  -0.643855427672130  -0.316610700189456  -0.065796481423312];
b = [0.362116891431710   0.002550864549783   0.285469330190766   0.887339907560975];
a= [0.177750163723039  -0.098440885775696  -0.978583409107433   0.032996707772190];

A = qu2om(a);
B = qu2om(b);
C = qu2om(c);
D = qu2om(d);
M_1 = qu2om(qmult(a, qinv(b)));
M_2 = qu2om(qmult(d,qinv(c)));
qu2ax(M_1)
qu2ax(M_2)