%% =========================================================================
%  fix_channel_gen_par.m   ——  并行版信道生成（解决"太慢"的问题）
% -------------------------------------------------------------------------
%  为什么要并行：
%     实测 ramgeo1.5_atten.exe 单次调用约 11 秒。
%     fct=10 时每信道 71 次 → 单信道 13.4 分钟 → 100 信道串行 22 小时。
%     本机 12 逻辑核，用 8 个 worker 可压到约 2.8 小时。
%
%  原理：
%     RAM 程序读写的是【当前工作目录】下的 ramgeo.in / tl.line，
%     所以每个 worker 必须有自己的工作目录，否则会互相覆盖。
%     做法：每个 parfor 迭代建一个独立的临时目录，cd 进去算完再 cd 回来。
%     临时目录放在系统 temp 下（纯 ASCII 路径），避免中文路径导致 exe 调用失败。
%
%  用法：
%     MATLAB 里直接运行（不需要 cd，路径全是绝对路径）
%     结果写到 D:/project/信道产生/data_fixed/
%
%  参数：
%     N_chan  —— 要生成多少个信道
%     nWork   —— 并行 worker 数（建议 = 物理核数，本机 8）
%     fct     —— 频率步进，10 Hz（71 点）。调大更快但插值误差变大
% =========================================================================
clc; clear;

%% ---- 参数 ---------------------------------------------------------------
work   = 'D:/project/信道产生';
N_chan = 50;          % 要生成的信道数
nWork  = 8;           % 并行 worker 数
fct    = 10;          % 频率步进 [Hz]
Dist   = 20000;       % 传播距离 [m]

fs = 24000;  Len = 8000;  df = fs/Len;
fl = 798;    fh = 1500;

Vfre   = fl:fct:fh;                 % 稀疏频点
f_full = fl:df:fh;                  % 3 Hz 完整网格
bin    = round(f_full/df);          % = 266:500  ★与 fct 无关★

assert(bin(1)==266 && bin(end)==500, 'bin 映射异常');

save_dir = fullfile(work, 'data_fixed');
if ~exist(save_dir,'dir'); mkdir(save_dir); end

% 把 exe 复制到 ASCII 路径下，避免中文路径调用失败
exe_src  = fullfile(work, 'ramgeo1.5_atten.exe');
tmp_root = fullfile(tempdir, 'ram_par');
if ~exist(tmp_root,'dir'); mkdir(tmp_root); end
exe_asc  = fullfile(tmp_root, 'ramgeo1.5_atten.exe');
copyfile(exe_src, exe_asc);

addpath(work);   % Input_PE / SeaWater_attenu / WriteenvRam

fprintf('信道数 %d | 频率步进 %d Hz (%d 点/信道) | 并行 %d workers\n', ...
        N_chan, fct, numel(Vfre), nWork);
fprintf('预计耗时约 %.0f 分钟\n', N_chan*numel(Vfre)*11.3/nWork/60);

%% ---- 开并行池 -----------------------------------------------------------
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', nWork);
end
fprintf('并行池已就绪：%d workers\n', pool.NumWorkers);

%% ---- 并行生成 -----------------------------------------------------------
t0 = tic;
done_count = 0;
dq = parallel.pool.DataQueue;
afterEach(dq, @(~) tick());

parfor ii = 1:N_chan
    td  = tempname(tmp_root);
    mkdir(td);
    old = cd(td);
    cleanup = onCleanup(@() restore_dir(old, td));

    Depth = 70 + 10*randi([1,12]);
    [bath, ssp, p1, p3, p4] = Input_PE(Depth, Dist);
    sd = randi([15, Depth-20]);
    rd = sd + randi([-10,10]);
    Range = bath(1, end);

    Vpress = zeros(1, numel(Vfre));
    for mm = 1:numel(Vfre)
        freq = Vfre(mm);
        dr  = 1500/1600*4;
        dz  = 1500/freq/20;
        wa1 = SeaWater_attenu(freq/1000, 30, 0, 33, 7, 'Thorpe');
        wa  = wa1*1500/freq;
        WriteenvRam(freq, sd, rd, dr, dz, bath, ssp, p1, p3, p4, Range, 8, wa);
        system(exe_asc);
        t0f = load('tl.line');
        Vpress(mm) = t0f(end,3) + 1i*t0f(end,4);
    end

    Vfull = interp1(Vfre, real(Vpress), f_full, 'linear', 'extrap') + ...
            1i * interp1(Vfre, imag(Vpress), f_full, 'linear', 'extrap');

    F = zeros(1, Len);
    F(bin) = Vfull;
    H = real(ifft(F));

    [~, pk] = max(abs(H));
    if pk > 2500
        H = H([pk-2500:end, 1:pk-2501]);
    else
        H = H([Len-2500+1+pk:end, 1:Len-2500+pk]);
    end

    channel_H = H(:).';
    save(fullfile(save_dir, sprintf('channel_H_%d.mat', ii)), ...
         'channel_H', 'fs', 'Len', 'fl', 'fh');

    send(dq, ii);
end

fprintf('\n全部完成，总耗时 %.1f 分钟\n', toc(t0)/60);

%% ---- 自检 ---------------------------------------------------------------
fprintf('\n================ 自检 ================\n');
npass = 0;
for ii = 1:min(N_chan, 10)
    H = load(fullfile(save_dir, sprintf('channel_H_%d.mat', ii)), 'channel_H');
    P = abs(fft(H.channel_H(:), Len)).^2;  P = P(1:Len/2+1);
    E = cumsum(P)/sum(P);
    lo = find(E>=0.005,1)/Len;  hi = find(E>=0.995,1)/Len;
    ok = (lo > 0.025 && hi < 0.075);
    npass = npass + ok;
    fprintf('  channel_H_%-3d  99%% band = %.5f ~ %.5f   %s\n', ii, lo, hi, string(ok));
end
fprintf('通过 %d / %d\n', npass, min(N_chan,10));
fprintf('======================================\n');


%% ---- 局部函数 -----------------------------------------------------------
function tick()
    persistent n
    if isempty(n); n = 0; end
    n = n + 1;
    if mod(n, 5) == 0
        fprintf('  已完成 %d 个信道\n', n);
    end
end

function restore_dir(old, td)
    cd(old);
    try; rmdir(td, 's'); catch; end
end
