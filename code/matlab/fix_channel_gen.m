%% =========================================================================
%  fix_channel_gen.m   ——  修正版「信道冲激响应」生成脚本
% -------------------------------------------------------------------------
%  【作用】
%      替代 信道产生/untitled2.m，批量生成 channel_H_*.mat
%      （每个文件是一个水声信道的时域冲激响应，8000 点，fs=24000 Hz）
%
%  【修的是什么】
%      原脚本用  F(fl/fct : fh/fct) = Vpress  写入频域。
%      这里 fl/fct 被【直接当成 DFT 的 bin 索引】用，所以：
%          fct = 3   →  F(266:500)  →  通带 798 ~ 1500 Hz   ✓ 正确
%          fct = 10  →  F(80:150)   →  通带 237 ~ 447 Hz    ✗ 整体下移 3.33 倍
%      本脚本把 bin 索引改成由 fs 和 Len 决定，与 fct 彻底解耦：
%          bin = round((fl:df:fh)/df)   % df = fs/Len = 3 Hz，恒等于 266:500
%      这样你仍然可以用 fct=10 稀疏采样省 RAM 调用次数，
%      再插值回 3 Hz 网格，通带位置不会跑掉。
%
%  【怎么用】
%      1) 把本文件放到 E:/project/data/信道产生/ 目录下
%         （该目录必须同时有 Input_PE.m、SeaWater_attenu.m、
%           WriteenvRam.m、ramgeo1.5_atten.exe、tl.line）
%      2) MATLAB 里 cd 到该目录，直接运行本脚本
%      3) 结果写到  E:/project/data/信道产生/data_fixed/  （不会覆盖原来的 data/）
%
%  【预期耗时】fct=10 时每个信道调用 RAM 71 次（原代码 235 次）
% =========================================================================
clc; clear; close all;

%% ---- 0. 工作目录 ---------------------------------------------------------
work = 'E:/project/data/信道产生';
cd(work);
assert(exist('ramgeo1.5_atten.exe','file') == 2, ...
    '当前目录找不到 ramgeo1.5_atten.exe，请确认 work 变量指向 信道产生 目录');

%% ---- 1. 栅格参数 ---------------------------------------------------------
fs  = 24000;        % 采样率 [Hz]  —— 由 Len=8000 且 df=3Hz 反推，与原代码一致
Len = 8000;         % 冲激响应长度 [点]
df  = fs/Len;       % 频率分辨率 = 3 Hz
fl  = 798;          % 换能器下限 [Hz]
fh  = 1500;         % 换能器上限 [Hz]

fct    = 10;        % 频率步进 [Hz]：只决定「算多少个频点」，不影响通带位置
Dist   = 20000;     % 传播距离 [m]   （老师原代码是 100000，请确认是否有意改动）
N_chan = 100;       % 要生成多少个信道

Vfre   = fl:fct:fh;          % 稀疏频点，71 个
f_full = fl:df:fh;           % 完整频点，235 个（3 Hz 网格）
bin    = round(f_full/df);   % = 266:500   ★关键：与 fct 无关★

fprintf('RAM 调用次数：%d 次/信道（原代码 235 次）\n', numel(Vfre));
fprintf('bin 索引    ：%d ~ %d  →  通带 %.0f ~ %.0f Hz\n', bin(1), bin(end), f_full(1), f_full(end));
assert(bin(1) == 266 && bin(end) == 500, 'bin 映射异常，请检查 fs 与 Len');

save_dir = fullfile(work, 'data_fixed');
if ~exist(save_dir, 'dir'); mkdir(save_dir); end

%% ---- 2. 逐信道计算 -------------------------------------------------------
for ii = 1:N_chan

    Depth = 70 + 10*randi([1,12]);
    [bath, ssp, p1, p3, p4] = Input_PE(Depth, Dist);
    sd = randi([15, Depth-20]);
    rd = sd + randi([-10,10]);
    Range = bath(1, end);

    Vpress = zeros(1, numel(Vfre));
    for mm = 1:numel(Vfre)
        freq = Vfre(mm);
        fprintf('信道 %d/%d | 频率 %d Hz (%d/%d)\n', ii, N_chan, freq, mm, numel(Vfre));

        dr  = 1500/1600*4;
        dz  = 1500/freq/20;
        wa1 = SeaWater_attenu(freq/1000, 30, 0, 33, 7, 'Thorpe');
        wa  = wa1*1500/freq;

        WriteenvRam(freq, sd, rd, dr, dz, bath, ssp, p1, p3, p4, Range, 8, wa);
        !ramgeo1.5_atten

        t0 = load('tl.line');
        Vpress(mm) = t0(end,3) + 1i*t0(end,4);
    end

    %% ---- 3. 插值回 3 Hz 网格，再写入正确的 bin -------------------------
    % 注意：interp1 不支持复数，必须实部虚部分开插值
    Vfull = interp1(Vfre, real(Vpress), f_full, 'linear', 'extrap') + ...
            1i * interp1(Vfre, imag(Vpress), f_full, 'linear', 'extrap');

    F = zeros(1, Len);
    F(bin) = Vfull;              % ★ 索引与 fct 无关

    H = real(ifft(F));           % 实数冲激响应

    % 把主到达时刻搬到 index 2501，方便后续对齐（与原代码一致）
    [~, pk] = max(abs(H));
    if pk > 2500
        H = H([pk-2500:end, 1:pk-2501]);
    else
        H = H([Len-2500+1+pk:end, 1:Len-2500+pk]);
    end

    channel_H = H(:).';
    save(fullfile(save_dir, sprintf('channel_H_%d.mat', ii)), ...
         'channel_H', 'fs', 'Len', 'fl', 'fh');
    fprintf('  → 已保存 %s/channel_H_%d.mat\n', save_dir, ii);
end

%% ---- 4. 自检 -------------------------------------------------------------
H  = load(fullfile(save_dir, 'channel_H_1.mat'), 'channel_H');
P  = abs(fft(H.channel_H(:), Len)).^2;
P  = P(1:Len/2+1);
E  = cumsum(P) / sum(P);
lo = find(E >= 0.005, 1) / Len;
hi = find(E >= 0.995, 1) / Len;

fprintf('\n================ 自检（channel_H_1.mat）================\n');
fprintf('99%% 能量频带 = %.5f ~ %.5f   （目标 0.0333 ~ 0.0625）\n', lo, hi);
if lo > 0.025 && hi < 0.075
    fprintf('✓ 通带位置正确\n');
else
    fprintf('✗ 通带位置仍然不对，请检查 fs / Len / bin 映射\n');
end
fprintf('=======================================================\n');
