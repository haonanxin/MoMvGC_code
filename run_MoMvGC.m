clear;clc;close all;
addpath('multi_datasets')
addpath('funs')

dataset_name='Caltech101-7';

load([dataset_name,'.mat'])

for i = 1 :length(X)
    X{i} = full((X{i} - mean(X{i}, 2)) ./ repmat(std(X{i}, [], 2), 1, size(X{i}, 2)));
end
num=size(X{1},1);
V=length(X);
c=length(unique(Y));

K=[2];
mu=10.^[1];
lambda=2^10;
beta=10.^[-3];
results=[];
for i3=1:length(K)
    A = constructA_vd(X, 5, K(i3));
    for i1=1:length(mu)
        for i2=1:length(beta)
            result=struct();
            [S,obj,H] = solver(X,num,V,mu(i1),lambda,beta(i2),K(i3),c,A);
            S(S<1e-5)=0;
            [clusternum1, y_learned]=graphconncomp(sparse(S));
            final = y_learned';
            MoMvGC_result = ClusteringMeasure_new(Y,final);
            result.K=K(i3);
            result.mu=mu(i1);
            result.beta=beta(i2);
            result.out=MoMvGC_result;
            result.H=H;
            results=[results,result];
        end
    end
end
disp(['********************************************']);
disp(['Running MoMvGC on ',dataset_name,' to obtain ACC: ', num2str(MoMvGC_result.ACC)]);
disp(['********************************************']);

