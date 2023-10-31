clear, clc, close all, format long
tic

   %  u_exact = double(imread('goldhill512.png')); 
     u_exact = double(imread('cameraman.tif'));
%     u_exact = double(imread('moon.tif'));
 %     u_exact = double(imread('kids.tif')); 
  
%  load ref2_512_MG_LENA; u_exact = img;
%    load ref2_512_MG_PEPPERS; u_exact = img;
umax = max(max(u_exact));
%u_exact = u_exact/umax;
%N=size(u_exact,1); kernel=ke_gen(N,300,10);




nxy =64; nx = nxy; ny = nxy; % Resize to reduce Problem
kernel=fspecial('gaussian',[nx/2 nx/2],1);



u_exact=imresize(u_exact,[nx nx]); kernel=imresize(kernel,[nx nx]);     
hx = 1 / nx; hy = 1 / ny; N=nx; hx2 = hx^2;
%  Extend kernel and compute its 2-d Fourier transform. Then use this to 
%  compute K'*z and kstark_hat, the 2-d Fourier transform for K'*K. 
kernel=kernel/sum(kernel(:));
m2 = 2*nx; nd2 = nx / 2; kernele = zeros(m2, m2) ;
kernele(nd2+1:nx+nd2,nd2+1:nx+nd2) = kernel ; %extention kernel
k_hat = fft2(fftshift(kernele)) ; clear kernele
z = integral_cgm(u_exact,k_hat,nx,nx);  % Blur Only  PLUS NOISE if needed
Zpsnr = psnr(z,u_exact)
beta =1;  alpha = 8e-9;  n = nx^2;  m = 2*nx*(nx-1);  nm = n + m;
computeB;    u0 = zeros(nx,nx);  U = u0;
M=speye(n,n);
tol = 1e-8; maxit = 1000; 
% u = z;
fprintf('iter    psnr\n')
fprintf('----    ----\n')

 figure;   imagesc(u_exact); 
 colormap(gray);
%  s=sprintf('exact image');s=title(s);  
size(u_exact)
 %zpsnr = psnr(z,u_exact);
 %fprintf('%d      %11.9g\n',0,zpsnr)
 figure;  imagesc(z); 
 colormap(gray);
%  ss=sprintf('blured image psnr = %0.5g',zpsnr);ss=title(ss);   
return
% ------- Parameters  ---------
c1 = 9.5e-7;  
c2 = 1e-6;
c3 = 1e-8;  
c4 = 1e-5;

d = 1.01; 
p = 2;
xeps = 0.00;
wgh = 1;

b0 = integral_cgm(z,conj(k_hat),nx,nx); b0 = b0(:);

% u = z;%Initial guess
u = zeros(nx,nx);

Q = zeros(nx,nx);

Px = zeros(nx,nx); 
Py = zeros(nx,nx);
Pz = zeros(nx,nx);

Nx = zeros(nx,nx); 
Ny = zeros(nx,nx);
Nz = zeros(nx,nx);

Mx = zeros(nx,nx); 
My = zeros(nx,nx);
Mz = zeros(nx,nx);

%Lagrange multipliers
lam_1 = zeros(nx,nx);

lamx_2 = zeros(nx,nx); 
lamy_2 = zeros(nx,nx);
lamz_2 = zeros(nx,nx);

lam_3 = zeros(nx,nx); 

lamx_4 = zeros(nx,nx); 
lamy_4 = zeros(nx,nx);
lamz_4 = zeros(nx,nx);

bete = 1e-8;
c_hat = fft2(kernel, nx, nx);
R =[];IT=[];
for k=1:5

[DerT1] = DerX(c2*Px+lamx_2,nx);
[DerT2] = DerY(c2*Py+lamy_2,nx);

b = b0 - wgh*hx2*DerT1(:) - wgh*hx2*DerT1(:);

L = -B'*B;
     [U,flag,rr,iter,rv] = pcg(@(x)KKLCH(nx,x,k_hat,L,alpha,c2),b,tol,maxit,[],[],u(:));
%PALM
%   [U,flag,rr,iter,rv] = pcg(@(x)KKLCH(nx,x,k_hat,L,alpha,c2),b,tol,maxit,@(x)PALM(nx,x,c_hat,100,1));
%  [U,flag,rr,iter,rv] = pcg(@(x)KKLCH(nx,x,k_hat,L,alpha,c2),b,tol,maxit,alpha*M+bete*diag(diag(L)));
%  [U,flag,rr,iter,rv] = pcg(@(x)KKLCH(nx,x,k_hat,L,alpha,c2),b,tol,maxit,M,diag(diag(L)),u(:));
%     [U,flag,rr,iter,rv] = pcg(@(x)KKLCH(nx,x,k_hat,L,alpha,c2),b,tol,maxit,@(x)PALMM(nx,x,c_hat,1000,10));
 

R(k)=rr;
IT(k)=k;

u = reshape(U,nx,nx); Upsnr = psnr(u,u_exact);

fprintf('%d      %11.9g\n',k,Upsnr)

[G1,G2] = Grad(u,nx);
TPx = G1 - lamx_2/c2 + ((c1 + lam_1)*Mx)/c2; 
TPy = G2 - lamy_2/c2 + ((c1 + lam_1)*My)/c2;
TPz = G2 - lamz_2/c2 + ((c1 + lam_1)*Mz)/c2;
Px = max( 0 , 1 - ((c1 + lam_1)./(c2*abs(TPx))))*TPx;
Py = max( 0 , 1 - ((c1 + lam_1)./(c2*abs(TPy))))*TPy;
Pz = max( 0 , 1 - ((c1 + lam_1)./(c2*abs(TPz))))*TPz;

[DerNx] = DerX(Nx,nx);
[DerNy] = DerY(Ny,nx);
[DerNz] = DerZ(Nz,nx);
TQ = DerNx + DerNy + DerNz - lam_3/c3;
Q = max( 0 , 1 - (alpha./(c3*abs(TQ))))*TQ;

[RHSx1] = DerX(c3*Q+lam_3,nx);
[RHSx21] = DerX(Nx,nx);
[RHSx22] = DerY(Ny,nx);
[RHSx2] = DerX(RHSx21+RHSx22,nx);
Nx = Mx - lamx_4/c4 - RHSx1/c4 + c3*RHSx2/c4;

[RHSy1] = DerY(c3*Q+lam_3,nx);
[RHSy2] = DerY(RHSx21+RHSx22,nx);
Ny = My - lamy_4/c4 - RHSy1/c4 + c3*RHSy2/c4;

Nz = Mz - lamz_4/c4;

Mx = Nx + lamx_4/c4 + ((c1 + lam_1)*Px)/c4; 
My = Ny + lamy_4/c4 + ((c1 + lam_1)*Py)/c4;
Mz = Nz + lamz_4/c4 + ((c1 + lam_1)*Pz)/c4;
if norm([Mx;My;Mz]) > 1
    Mx = Mx./norm([Mx;My;Mz]);
    My = My./norm([Mx;My;Mz]);
    Mz = Mz./norm([Mx;My;Mz]);
end
%Lagrange multipliers update
lam_1 = lam_1 + c1*((abs(Px)+abs(Py)+abs(Pz)) - (Px.*Mx + Py.*My + Pz.*Mz));

lamx_2 = lamx_2 + c1*(Px - G1);
lamy_2 = lamy_2 + c1*(Py - G2);
lamz_2 = lamz_2 + c1*(Pz - G2);

lam_3 = lam_3 + c3*(Q - DerNx - DerNy - DerNz);

lamx_4 = lamx_4 + c4*(Nx - Mx);
lamy_4 = lamy_4 + c4*(Ny - My);
lamz_4 = lamz_4 + c4*(Nz - Mz);

end
toc
figure;  imagesc(u); colormap(gray);
% ss=sprintf('k = %d , psnr = %0.5g',k,Upsnr);ss=title(ss);

figure
hold on 
plot(IT,R)
%hold off
xlabel('iteration number')
ylabel('relative residuals')














