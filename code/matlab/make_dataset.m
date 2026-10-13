function make_dataset(cfg)
%MAKE_DATASET  生成训练/验证/测试数据集（修正版）
%==========================================================================
%  【任务定义】去噪，不是解卷积
%      干净信号 clean = conv(源chirp, 信道H)     ← 无噪的接收波形
%      含噪信号 noisy = clean + 高斯噪声
%      训练目标       = clean
%
%  【为什么这样改】
%      旧做法把「源 chirp」当目标，等于要求模型在不知道信道的情况下反卷积，
%      是病态问题，模型只能靠背波形作弊。改成「无噪接收波形」当目标后，
%      目标随信道和 chirp 变化，且是可实现的（不需要信道估计）。
%
%  【多样性来源】四层相乘
%      1) 信道       data_fixed/channel_H_*.mat
%      2) 源 chirp   参数化生成（起止频率/时长/扫频方向/包络/初相/幅度）
%      3) 信噪比     多档
%      4) 噪声实现   每档多次
%
%  【划分方式】按 chirp 分组
%      测试集的 chirp 从没在训练里出现过 —— 这才是真的泛化测试。
%      随机按样本划分是假的，因为同一个 chirp 会同时出现在训练和测试里。
%
%  【用法】
%      make_dataset                       % 用默认配置
%      make_dataset(struct('n_noise',2))  % 覆盖个别参数
%==========================================================================

if nargin < 1 || isempty(cfg); cfg = struct(); end

def = struct( ...
    'ch_dir',        'D:/project/信道产生/data_fixed', ...
    'out_root',      'D:/project/dataset', ...
    'fs_chirp',      9600, ...        % 源信号原始采样率
    'fs_ch',         24000, ...       % 信道采样率
    'n_chirp_train', 14, ...
    'n_chirp_val',   3, ...
    'n_chirp_test',  3, ...
    'n_noise',       4, ...           % 每个(信道,chirp,SNR)组合的噪声实现数
    'snr_list',      [-10 -5 0 5 10], ...
    'modtype',       'LFM', ...       % 'LFM' 线性调频 | 'HFM' 双曲调频
    'normalize',     true, ...        % 把 clean 归一化到单位 rms（模拟 AGC）
    'seed',          20261013);
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(cfg, fn{k}); cfg.(fn{k}) = def.(fn{k}); end
end
rng(cfg.seed, 'twister');

%% ---- 1. 信道清单 --------------------------------------------------------
files = dir(fullfile(cfg.ch_dir, 'channel_H_*.mat'));
assert(~isempty(files), '在 %s 找不到 channel_H_*.mat', cfg.ch_dir);

% ---- 内容去重 -----------------------------------------------------------
% 注意：MATLAB 的 .mat 文件头带创建时间戳，两个内容完全相同的文件 MD5 也不同，
%       所以【不能用 MD5 判重】，必须逐元素比较。
Hmat = []; keep = [];
for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name), 'channel_H');
    v  = S.channel_H(:).';
    if isempty(Hmat) || ~any(all(abs(Hmat - v) < 1e-30, 2))
        Hmat(end+1,:) = v;  %#ok<AGROW>
        keep(end+1)   = k;  %#ok<AGROW>
    end
end
n_raw = numel(files);
files = files(keep);
if n_raw > numel(files)
    fprintf('信道去重：%d -> %d（剔除 %d 个内容重复的）\n', ...
            n_raw, numel(files), n_raw - numel(files));
end

n_ch = numel(files);
fprintf('信道：%d 个（来自 %s）\n', n_ch, cfg.ch_dir);

%% ---- 2. 生成 chirp 参数表 ----------------------------------------------
% 老师的 chirp_signal.mat 实测参数（已用 gen_chirp 精确复现，corr=1.000000）：
%     LFM，798 -> 1500 Hz，T = 1.0 s，恒幅，幅值 ±1，fs = 9600
% 下面第 1 组就取这个参数，其余在该参数附近扰动 —— 既覆盖老师的信号，
% 又有足够多样性。
n_chirp = cfg.n_chirp_train + cfg.n_chirp_val + cfg.n_chirp_test;
P = struct('f0',{},'f1',{},'T',{},'env',{},'phi',{},'A',{});

P(1).f0 = 798;  P(1).f1 = 1500;  P(1).T = 1.0;
P(1).env = 'rect';  P(1).phi = 0;  P(1).A = 1;

for i = 2:n_chirp
    a = 800  + 250*rand();            % 低频端 800 ~ 1050 Hz
    b = 1280 + 219*rand();            % 高频端 1280 ~ 1499 Hz
    if rand() < 0.5
        P(i).f0 = a;  P(i).f1 = b;    % 上扫（与老师一致）
    else
        P(i).f0 = b;  P(i).f1 = a;    % 下扫
    end
    P(i).T = 0.6 + 0.4*rand();        % 0.6 ~ 1.0 s

    r = rand();                       % 包络：偏向恒幅（老师用的是恒幅）
    if r < 0.5
        P(i).env = 'rect';
    elseif r < 0.75
        P(i).env = 'hann';
    else
        P(i).env = 'gauss';
    end

    P(i).phi = 2*pi*rand();
    P(i).A   = 0.5 + 0.5*rand();
    assert(min(P(i).f0,P(i).f1) >= 800 && max(P(i).f0,P(i).f1) <= 1500, ...
        'chirp %d 频带越界（必须落在信道通带 798~1500 Hz 内）', i);
end

split.chirp_id = {1:cfg.n_chirp_train, ...
                  cfg.n_chirp_train + (1:cfg.n_chirp_val), ...
                  cfg.n_chirp_train + cfg.n_chirp_val + (1:cfg.n_chirp_test)};
split.name     = {'train','val','test'};

fprintf('chirp：%d 组（train %d / val %d / test %d）\n', n_chirp, ...
        cfg.n_chirp_train, cfg.n_chirp_val, cfg.n_chirp_test);
fprintf('SNR 档位：%s dB\n', mat2str(cfg.snr_list));

n_per_split = [numel(split.chirp_id{1}), numel(split.chirp_id{2}), numel(split.chirp_id{3})] ...
              * n_ch * numel(cfg.snr_list) * cfg.n_noise;
fprintf('\n预计样本数：train %d / val %d / test %d，合计 %d\n', ...
        n_per_split(1), n_per_split(2), n_per_split(3), sum(n_per_split));
L = round(P(1).T*cfg.fs_chirp)*(cfg.fs_ch/cfg.fs_chirp) + 8000 - 1;
fprintf('单样本约 %d 点，磁盘约 %.2f GB\n\n', round(L), sum(n_per_split)*L*4*2/1e9);

%% ---- 3. 逐 split 生成 ---------------------------------------------------
for si = 1:3
    outdir = fullfile(cfg.out_root, split.name{si});
    if ~exist(outdir,'dir'); mkdir(outdir); end
    delete(fullfile(outdir,'*.mat'));

    meta = struct('chirp_id',{},'chan_id',{},'snr',{},'file',{}, ...
                  'f0',{},'f1',{},'T',{},'env',{});
    cnt = 0; t0 = tic;

    for ci = split.chirp_id{si}
        p = P(ci);
        s = gen_chirp(cfg.fs_chirp, p.T, p.f0, p.f1, cfg.modtype, p.env, p.phi, p.A);
        s = resample(s, 5, 2);                  % 9600 -> 24000
        s = s / max(abs(s));

        for hi = 1:n_ch
            S = load(fullfile(files(hi).folder, files(hi).name), 'channel_H');
            h = S.channel_H(:).';

            clean_signal = conv(s, h);
            if cfg.normalize
                clean_signal = clean_signal / rms(clean_signal);
            end
            Pc = mean(clean_signal.^2);

            for snr = cfg.snr_list
                Pn = Pc / 10^(snr/10);
                for rep = 1:cfg.n_noise
                    noisy_signal = clean_signal + randn(size(clean_signal)) * sqrt(Pn);
                    cnt = cnt + 1;
                    fname = sprintf('s_%05d.mat', cnt);
                    true_snr  = snr;
                    chirp_id  = ci;
                    chan_id   = hi;
                    save(fullfile(outdir, fname), ...
                         'noisy_signal','clean_signal','true_snr','chirp_id','chan_id');
                    meta(end+1) = struct('chirp_id',ci,'chan_id',hi,'snr',snr, ...
                        'file',fname,'f0',p.f0,'f1',p.f1,'T',p.T,'env',p.env); %#ok<AGROW>
                end
            end
        end
        fprintf('  [%s] chirp %2d/%2d (%.0f-%.0f Hz, %.2f s, %s) 累计 %d 样本  %.1f min\n', ...
            split.name{si}, ci, n_chirp, p.f0, p.f1, p.T, p.env, cnt, toc(t0)/60);
    end

    meta = meta(1:cnt);
    save(fullfile(cfg.out_root, sprintf('meta_%s.mat', split.name{si})), 'meta', 'P', 'cfg');
    fprintf('  [%s] 完成：%d 样本 -> %s\n\n', split.name{si}, cnt, outdir);
end

save(fullfile(cfg.out_root, 'chirp_params.mat'), 'P', 'split', 'cfg');
fprintf('全部完成。参数表 -> %s/chirp_params.mat\n', cfg.out_root);
fprintf('注意：样本里同时存了 noisy_signal 和 clean_signal，\n');
fprintf('      Python 读取时必须【按变量名取】，不能取「第一个非 __ 键」。\n');
end
