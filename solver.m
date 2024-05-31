function [S,obj,H] = solver(X,num,V,mu,lambda,beta,K,c,A)
zr=1e-11;
% [A] = gen_high_order_graph(X,K);
H=ones(V,K)./(V*K);
SS=0;
% for k=1:K
    for v=1:V
        SS=SS+A{v,1};
    end
% end

SS=(SS+SS')/2;
L=diag(sum(SS,2))-SS+eye(num)*eps;
[F, temp, ev]=eig1(L,c+1, 0);
F=F(:,2:c+1);
% F=zeros(num,c);
% F(sub2ind(size(F),1:num,Y'))=1;
F = F./repmat(sqrt(sum(F.^2,2)),1,c);

iter=50;
for t=1:iter
    % update S
    distance= pdist(F);
    D=lambda*squareform(distance.^2);
    B=zeros(num,num);
    for v =1:V
        for k=1:K
            B=B+H(v,k)*A{v,k};
        end
    end
    M=D-2*B;
    S=zeros(num,num);
    for i=1:num
        tempS=EProjSimplex_new(-M(i,:)/(2*mu+2));
        S(i,:)=tempS';
    end

    % update H
    R=zeros(V*K,V*K);
    P=zeros(V,K);
    for v=1:V
        R(K*(v-1)+1:K*v,K*(v-1)+1:K*v)=ones(K,K);
        for k=1:K
            P(v,k)=sum(sum((S-A{v,k}).^2));
        end
    end
    R=0*R+diag(ones(V*K,1));
    Q=reshape(P',[],1);
    H_ba = quadprog(2*beta*R,Q,[],[],ones(1,K*V),1,zeros(K*V,1),ones(K*V,1),[], optimset('Display', 'off'));
    tempH=reshape(H_ba,[K,V]);
    H=tempH';
    obj1=reshape(P,1,[])*reshape(H,[],1);

    % updata F
    S=(S+S')/2;
    L=diag(sum(S,2))-S+eye(num)*eps;
    F_old=F;
    [F, temp3, ev]=eig1(L,c, 0);
    obj(t)=obj1+beta*H_ba'*R*H_ba+mu*trace(S'*S)+lambda*trace(F'*L*F);
    fprintf('iter:%d\n',t);

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > zr
        lambda = 2*lambda;
    elseif fn2 < zr
        lambda = lambda/2;
        F = F_old;
    else
        break;
    end


end
% 
% figure(1);
% plot(1:t,obj(1:t))
end