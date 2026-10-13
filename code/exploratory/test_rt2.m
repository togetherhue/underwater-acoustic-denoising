%% 正确的往返测试：用合成 Vpress 验证抢救链路
clc; clear; rng(1);
fs=24000; Len=8000; df=fs/Len; fl=798; fh=1500;
Vfre = fl:10:fh;            % 798..1498, 71 个（和旧脚本一致）
f_full = fl:df:fh;          % 798..1500, 235 个
bin = round(f_full/df);     % 266..500

% ---- 1. 合成一个传递函数（71 个复数，量级与真实信道接近）----
Vpress = (randn(1,71) + 1i*randn(1,71)) .* exp(-(0:70)/25) * 1e-4;

% ---- 2. "错误信道"：放到 bin 80:150（复刻旧脚本的 bug）----
F_old = zeros(1,Len); F_old(80:150) = Vpress;
h_old = real(ifft(F_old));
h_old = shift2501(h_old, Len);

% ---- 3. "正确信道"：插值后放到 bin 266:500（复刻修正脚本）----
Vfull = interp1(Vfre, real(Vpress), f_full,'linear','extrap') + ...
        1i*interp1(Vfre, imag(Vpress), f_full,'linear','extrap');
F_ref = zeros(1,Len); F_ref(bin) = Vfull;
h_ref = real(ifft(F_ref));
h_ref = shift2501(h_ref, Len);

% ---- 4. 抢救 ----
Fb  = fft(h_old(:), Len);
Vr  = Fb(80:150).' * 2;          % 取出 71 个值，×2 补偿 real()
Vf2 = interp1(Vfre, real(Vr), f_full,'linear','extrap') + ...
      1i*interp1(Vfre, imag(Vr), f_full,'linear','extrap');
Fc  = zeros(1,Len); Fc(bin) = Vf2;
h_res = real(ifft(Fc));
h_res = shift2501(h_res, Len);

% ---- 5. 对比 ----
cc = corrcoef(h_ref(:), h_res(:));
fprintf('=== 正确往返测试 ===\n');
fprintf('h_ref peak@%d  max=%.4e  rms=%.4e\n', find(abs(h_ref)==max(abs(h_ref)),1), max(abs(h_ref)), rms(h_ref));
fprintf('h_res peak@%d  max=%.4e  rms=%.4e\n', find(abs(h_res)==max(abs(h_res)),1), max(abs(h_res)), rms(h_res));
fprintf('相关系数 = %.8f\n', cc(1,2));
fprintf('相对误差 = %.8f\n', norm(h_ref(:)-h_res(:))/norm(h_ref(:)));

% 检查提取是否精确
fprintf('\n提取精度检查：\n');
fprintf('  ||Vr - Vpress||/||Vpress|| = %.3e\n', norm(Vr(:)-Vpress(:))/norm(Vpress(:)));

% 抢救后信道的频带
P = abs(fft(h_res(:), Len)).^2; P = P(1:Len/2+1);
E = cumsum(P)/sum(P);
fprintf('\n抢救后 99%% 频带 = %.5f ~ %.5f  （目标 0.0333~0.0625）\n', ...
        find(E>=0.005,1)/Len, find(E>=0.995,1)/Len);

% 对比：抢救后 vs 直接卷积出的正确信道，误差来源
fprintf('\n=== 结论 ===\n');
if cc(1,2) > 0.99
    fprintf('抢救链路无损，可以放心用于 240 个旧信道。\n');
else
    fprintf('抢救链路有损，需要重跑 RAM。\n');
end


function y = shift2501(x, Len)
x = x(:).';
[~,pk]=max(abs(x));
if pk>2500; y=x([pk-2500:Len, 1:pk-2501]);
else;       y=x([Len-2500+1+pk:Len, 1:Len-2500+pk]); end
end
