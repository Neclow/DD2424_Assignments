function [X, Y, y] = LoadBatch(filename)
%Reads data from CIFAR-10 batch file
%Out: image and label data in separate files
%X: image pixel data, size d x N
%Y: One-hot representation of the label for each image, size K x N
%y: label for each image, size N x 1
%N = nb of images (10000), d = dimensionality of each image (32x32x3)
%K = nb of labels (10)

A = load(filename);
X = double(A.data')./255;
y = double(A.labels+1);
Y = eye(max(y));
Y = Y(y,:)';
end