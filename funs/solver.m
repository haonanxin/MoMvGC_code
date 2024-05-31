function [S,obj] = solver(NE,d_feature,S,f_A,X,A,d,num,numclass,lambda,beta,zr,F)
obj=zeros(NE,1);
for u=1:NE
    % update W
    if d_feature>numclass
        temp_M=S*f_A*X;
        M=temp_M'*A*temp_M;
        [V, temp1, ev1]=eig1(M,d, 1);
        W=V;
    else
        W=eye(d_feature);
    end


    distance= pdist(F);
    K=lambda*squareform(distance.^2);

    % update S
    temp_B=f_A*X*W;
    B=temp_B*temp_B';

    %             [S] = update_S_ALM(S,B,K,A,num,beta);
    [S] = update_S_QP_new(S,B,K,A,num,beta);
    S=full(S);
    % update F
    S=(S+S')/2;
    L=diag(sum(S,2))-S+eye(num)*eps;
    F_old=F;
    [F, temp, ev]=eig1(L,numclass, 0);
    fn1 = sum(ev(1:numclass));
    fn2 = sum(ev(1:numclass+1));

    fprintf('iter:%d\n',u);
    obj(u)=-trace(S*B*S'*A)+2*lambda*trace(F'*L*F)+beta*trace(S'*S);

    if fn1 > zr
        lambda = 2*lambda;
    elseif fn2 < zr
        lambda = lambda/2;
        F = F_old;
    else
        break;
    end


end


end

