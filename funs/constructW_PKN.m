% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function W = constructW_PKN(X, k, issymmetric)
% X: each column is a data point
% k: number of neighbors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

if nargin < 3
    issymmetric = 1;
end
if nargin < 2
    k = 5;
end

n = size(X, 1);
D = squareform(pdist(X, 'squaredeuclidean'));

W = sparse(n, n);

[D, idx] = mink(D, k + 2, 2);
% [D,idx] = temp_mink(D, k + 2);

D = D(:, 2:end);
idx = idx(:, 2:end);

row = repmat(1:n, 1, k);
col = reshape(idx(:, 1:k), 1, []);
ind = sub2ind(size(W), row, col);

W(ind) = (D(:, k + 1) - D(:, 1:k)) ./ (k * D(:, k + 1) - sum(D(:, 1:k), 2));

if issymmetric == 1
    W = (W+W')/2;
end
end

function [final_val,final_index] = temp_mink(A, num)
[m,n]=size(A);
val=zeros(m,n);
index=zeros(m,n);
for j=1:m
    [M,I]=sort(A(j,:),'ascend');
    val(j,:)=M';
    index(j,:)=I';
end
final_val=val(:,1:num);
final_index=index(:,1:num);
end