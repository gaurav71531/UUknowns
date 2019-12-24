% Reference: (1) Wavelet Analysis of Fractional Brownian Motion
%            
% To functionalize the code to accept arguments for estimation of
% fractional differencing orders individually for each time series.
%

function [d1]=WT_estimator_v4(series,series_num)

Wt = HaarWaveletTransform;

N = length(series);           % Number of sample points
K = series_num;              % Number of time series
X = zeros(K,N);

for i = 1 : K
    X(i,:) = series(i,:);
    mean = Wt.Mean(X(i,:));
    X(i,:) = X(i,:) - mean;
end


NumScales = floor(log2(N));
log_wavelet_scales = zeros (1, NumScales);

% subplot(2,1,1);
% plot (X(1,:));
% hold on;
% plot (X(2,:));
% hold on;
% plot (X(3,:));
% hold off;
% title('Input series','Fontsize',20);
% xlabel('Record ID','Fontsize',20);


% subplot(4,1,3);
% plot ( X(2,:));
% title('Input series','Fontsize',20);
% xlabel('Record ID','Fontsize',20);

% Wavelet analysis
scale = 1:NumScales;

% % % % patch for zero variance log_scales
% work around: set the log_scale to previous value, as constant signals
% will have zero variance here, and we can make log_scales constant in that
% part to have fractal coeff close to zero

epsilon = 1e-10;

% To keep uniform with 'R'.
% V = Scales Coefficients.
% W = Wavelet Coefficients.
[~, W] = Wt.Transform(X(1,:));
% W1 = haarWavelet(X(1,:));
% [a, d] = haart(X(1,:));
j = floor(N / 2);                  % Represents the num coefficients
for i = 1: (NumScales - 1)          % The last scale has one coeff.only.
    y = W(i,1:j);
    %variance = Wt.varianceEst(y);
    variance = var(y);
    if variance < epsilon
        log_wavelet_scales(i) = log_wavelet_scales(i-1);
    else
        log_wavelet_scales(i) = log2(variance); 
    end
    j = floor(j/2);
end
% subplot(2,1,2);
% plot(scale(1:(NumScales - 1)),log_wavelet_scales(1:(NumScales - 1)),'o');
% 
% % % % % % % % % % % % % % % % % % % % % % % % % 
% % Linear Fit to get the slope
% % of the Log2-Log2 relation
% % "Wavelet Analysis and Synthesis
% % of Fractional Brownian Motion"

numScaleThres = 7;
% numScaleThres =NumScales-1;
p = polyfit(scale(1:numScaleThres), log_wavelet_scales(1:numScaleThres), 1);
H = (p(1)-1)/2;

% xUse = scale(1:(NumScales - 1));
% yLin = @(x) p(1)*x + p(2);
% figure;plot(scale(1:(NumScales - 1)), log_wavelet_scales(1:(NumScales - 1)));
% hold on;
% plot(xUse, yLin(xUse), 'r');
% brkPntInd = findchangepts(log_wavelet_scales(1:(NumScales - 1)), 'maxNumChanges', 1, 'Statistic', 'linear');
% scatter(brkPntInd, log_wavelet_scales(brkPntInd), 'filled');
% grid
% d1= H;
% % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % % % % % % % % % % % % % 
% Fractional derivatives of random walks: Time 
% series with long-time memory
d1 = 1/2 - H;

% % % % % % % % % % % % % % % % % % % % % % % % % %


% fit with 2 piece-wise linear functions
% x = scale(1:(NumScales - 1));
% y = log_wavelet_scales(1:(NumScales - 1));
% brkPntInd = findchangepts(y, 'maxNumChanges', 1);
% 
% p1 = polyfit(x(1:brkPntInd), y(1:brkPntInd), 1);
% p2 = polyfit(x(brkPntInd+1:end), y(brkPntInd+1:end), 1);
% 
% d1 = [p1(1)/2, p2(1)/2];






% % 
% % Use coefficients from ployfit to estimate
% % a new set of "Y" values
% Y = polyval(p, scale(1:(NumScales - 1)));
% hold on;
% plot(scale(1:(NumScales - 1)), Y);
% title('Estimated fractional orders@All channels','Fontsize',24);
% xlabel('Wavelet Scale number','Fontsize',24);
% ylabel('Log(Var(Wave-coeffs))')

function W = haarWavelet(x)

N = length(x);
j = N;

w = zeros(1,N);
for i = 1:floor(log2(N))
    j = floor(j/2);
    w(1:j) = (x(1:2:2*j-1) + x(2:2:2*j))/sqrt(2);
    w(j+1:2*j) = (x(1:2:2*j-1) - x(2:2:2*j))/sqrt(2);
    
    x(1:2*j) = w(1:2*j);
end



