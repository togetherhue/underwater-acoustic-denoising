%% 验证：能不能直接从旧 channel_H 里把正确信道"抢救"出来，不用重跑 RAM
%
% 推理：旧脚本 F(fl/fct:fh/fct)=Vpress 把 fct=10 时的 71 个值
%       放到了 bin 80:150（对应 240~450Hz）。
%       但这 71 个值本身是 RAM 在 798:10:1500 Hz 算出来的，物理上是对的，
%       只是被放错了频率位置。
%       所以只要把它们从 bin 80:150 取出来，插值后放回 bin 266:500 即可。

clc; clear;
old_file = 'D:/project/信道产生/data/channel_H_1.mat';
fs = 24000; Len = 8000; df = fs/Len;
fl = 798; fh = 1500; fct = 10;

S = load(old_file, 'channel_H');
h_old = S.channel_H(:).';

%% 1. 找出旧信道实际占了哪些 bin
F = fft(h_old(:), Len);
mag = abs(F(1:Len/2+1));
thr = max(mag) * 1e-3;
idx = find(mag > thr);
fprintf('旧信道能量显著的 bin：%d ~ %d  （共 %d 个）\n', idx(1), idx(end), numel(idx));
fprintf('对应频率（按 fs=24000）：%.1f ~ %.1f Hz\n', (idx(1)-1)*df, (idx(end)-1)*df);

%% 2. 取出这 71 个值（乘 2 补偿 real() 造成的幅度减半）
n = numel(idx);
Vpress_rec = F(idx) * 2;
fprintf('\n取出的传递函数个数 = %d（期望 %d）\n', n, numel(fl:fct:fh));
fprintf('|Vpress| 前 3 个 = %s\n', mat2str(abs(Vpress_rec(1:3))', 4));

%% 3. 插值回 3 Hz 网格，放回正确的 bin 266:500
Vfre   = fl:fct:fh;
f_full = fl:df:fh;
bin    = round(f_full/df);

% 旧 bin 与频率的对应关系：bin(idx(k)) 实际代表 Vfre(k)
Vfull = interp1(Vfre, real(Vpress_rec), f_full, 'linear', 'extrap') + ...
        1i * interp1(Vfre, imag(Vpress_rec), f_full, 'linear', 'extrap');

Fnew = zeros(1, Len);
Fnew(bin) = Vfull;
h_new = real(ifft(Fnew));

%% 4. 自检
P = abs(fft(h_new(:), Len)).^2;  P = P(1:Len/2+1);
E = cumsum(P)/sum(P);
lo = find(E>=0.005,1)/Len;
hi = find(E>=0.995,1)/Len;
fprintf('\n=== 抢救后的信道 ===\n');
fprintf('99%% 能量频带 = %.5f ~ %.5f   （目标 0.0333 ~ 0.0625）\n', lo, hi);
if lo > 0.025 && hi < 0.075
    fprintf('✓ 频带正确\n');
else
    fprintf('✗ 频带仍然不对\n');
end

% 与原修正版（重跑 RAM 得到的）对比
vfile = 'D:/project/信道产生/data_verify/channel_H_1.mat';
if exist(vfile, 'file')
    V = load(vfile, 'channel_H');
    h_ref = V.channel_H(:).';
    c = corrcoef(h_new(:), h_ref(:));
    fprintf('\n与重跑 RAM 得到的信道对比：相关系数 = %.4f\n', c(1,2));
    fprintf('（不同信道是随机的，这里只验证量级和形态是否一致）\n');
    fprintf('抢救后 rms = %.4e  max|h| = %.4e\n', rms(h_new), max(abs(h_new)));
    fprintf('重跑后 rms = %.4e  max|h| = %.4e\n', rms(h_ref), max(abs(h_ref)));
end

% 画出前 200 点看看形态
fprintf('\n抢救后信道前 20 点：\n');
fprintf('  %s\n', mat2str(h_new(1:20), 4));
