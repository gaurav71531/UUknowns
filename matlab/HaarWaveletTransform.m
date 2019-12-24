%----------------------------------------------------------------------- 
%  Discrete Wavelet Transform Class
%
% May-16-20-15  Started.                                    [Saul R.]
%-----------------------------------------------------------------------


classdef HaarWaveletTransform
   properties
      Value
   end
   
   methods
       
      %---------------------------------------------- 
      % Transform() - Returns a Matrix with Log2 different
      %               levels of coefficient scales
      %---------------------------------------------- 
      
      function [W, D] = Transform(obj, Signal)
        N = length(Signal);
        W = zeros(floor(log2(N)), floor(N/2));
        D = zeros(floor(log2(N)), floor(N/2));
        j = N;
        for i = 1: floor(log2(N)) 
            j = floor(j / 2);
            [w, d] = obj.dwthaar(Signal);
            W(i,1:j) = w;
            D(i, 1:j) = d;
            Signal = w;
        end
      end
      
    %----------------------------------------------   
    % One step Haar DWT,Ref: Ripples in Mathematics.
    % Modified to make it compatible with R.
    %----------------------------------------------   
    
    function [C, S] = dwthaar(obj, Signal)
        N = length(Signal);
        C = zeros(1, floor(N/2));
        S = zeros(1, floor(N/2));

        for n = 1:floor(N/2)
            C(n) = 1/2*(Signal(2*n-1) + Signal(2*n));
            S(n) = 1/2*(Signal(2*n-1) - Signal(2*n));
        end
        % This step is to make it compatible with R.
        C =  2 * C / sqrt(2);
%         S = -2 * S / sqrt(2);
        S =  2 * S / sqrt(2);
    end
    
    
    %----------------------------------------------
    %
    %----------------------------------------------
    
    function m = Mean(obj, Signal)
        N = length(Signal);
        t = ones(N,1);
        m = 1/N * (Signal*t);
    end
    
    
    %----------------------------------------------   
    % Variance estimator. x = wavelet coeffs of a 
    % specific "level" -assumes average = 0.
    %----------------------------------------------   
    
    function v = varianceEst(obj, x)
        N = length(x);
        v = 1/N * (x * x');
    end
    
   end     
end