function make_dataset_pam(noise_type, out_root, alpha_val, splits_override, seed_mode, burst_override, snr_override)
%MAKE_DATASET_PAM  按导师方案生成"非高斯脉冲噪声 + 突发干扰"数据集
%==========================================================================
%  信号模型（老师文档第一节）：
%      y_n = s_n + v_n + i_n
%      s : 期望水声信号（chirp 过 RAM 多径信道）
%      v : 海洋背景噪声 —— 'alphastable' 时用 α 稳定分布；'gauss' 时为高斯（对照）
%      i : 脉冲/突发干扰 —— 随机幅度、随机持续时间、随机突发间隔
%
%  【噪声强度定义】α 稳定分布（α<2）方差无穷，无法用功率定义信噪比。
%  本脚本采用【75% 分位数匹配】：令噪声的 q75(|v|) 等于同功率高斯噪声的
%  q75（= 0.6745×σ）。这样"核心宽度"一致，α 越小则尾部越重、脉冲性越强，
%  两类噪声的"典型幅度"可比。
%
%  【配对设计】背景噪声与干扰都用【按样本序号确定的种子】生成，因此
%  'alphastable' 与 'gauss' 两套数据的【突发干扰实现完全相同】，
%  只有背景噪声的分布不同 —— 构成配对对照。
%
%  信道/波形划分沿用 v3 约定（与既有全部结果可比）：
%      train   chirp 1-14  × 信道 1-18
%      val     chirp 15-17 × 信道 19-21
%      testA   chirp 18-20 × 信道 1-18     （新波形 + 见过的信道）
%      testC   chirp 18-20 × 信道 22-25    （新波形 + 新信道）
%
%  用法：make_dataset_pam('alphastable', 'D:/project/dataset_pam/alphastable', 1.5)
%        make_dataset_pam('gauss',       'D:/project/dataset_pam/gauss')
%        % 干扰与 chirp 解耦（用于隔离"波形效应"）：
%        make_dataset_pam('alphastable','D:/project/ds_scan_paired',1.5, sp, 'paired')
%==========================================================================

if nargin < 1 || isempty(noise_type); noise_type = 'alphastable'; end
if nargin < 2 || isempty(out_root);   out_root = fullfile('D:/project/dataset_pam', noise_type); end
if nargin < 3 || isempty(alpha_val);  alpha_val = 1.5; end
if nargin < 4; splits_override = []; end
if nargin < 5 || isempty(seed_mode); seed_mode = 'index'; end
if nargin < 6; burst_override = []; end
if nargin < 7 || isempty(snr_override); snr_override = []; end

cfg.fs_chirp     = 9600;
cfg.fs           = 24000;
cfg.n_chirp_total = 20;
cfg.snr_list      = [-10 -5 0 5 10];      % 名义 SNR（按 q75 匹配定义）
cfg.modtype       = 'LFM';
cfg.normalize     = true;
cfg.seed          = 20261013;
cfg.ch_dir        = 'E:/project/data/信道产生/data_fixed';
cfg.noise_type    = lower(noise_type);
cfg.alpha         = alpha_val;
cfg.beta          = 0;                     % 对称 α 稳定分布
cfg.Q75_GAUSS     = 0.6745;                % q75(|N(0,1)|)

% 可选：覆盖 SNR 列表（用于"等有效功率 SNR 的 α 重扫"）。
% 背景：q75 匹配让名义 SNR 固定，但 α<2 时方差无穷，有效功率 SNR 会随 α 大幅漂移
%   （实测 shift = 有效SNR − 名义SNR：α=1.5 为 +6.7 dB、α=1.2 为 −6.4 dB、
%    α=1.8 为 +16.7 dB、α=2.0 为 +20.8 dB）。若不做等功率重扫，
%   "α 扫描"实际同时改变了噪声强度与尾形，无法归因。
% 用法：make_dataset_pam('alphastable', root, 1.2, sp, 'index', [], [-10 -5 0 5 10]+13.07)
if ~isempty(snr_override)
    assert(isnumeric(snr_override) && isvector(snr_override), 'snr_override 必须是数值向量');
    cfg.snr_list = snr_override(:).';
end

% 噪声/干扰的随机数种子模式：
%   'index'  （默认）种子 = seed + 样本序号 cnt。由于 chirp 是最外层循环，
%            每个 chirp 独占一段连续种子区间 -> 【干扰实现与 chirp 序号混杂】，
%            此时逐 chirp 的差异无法归因于波形。
%   'paired' 种子只由 (信道, SNR, 重复) 决定，与 chirp 无关；并且把干扰/噪声
%            生成在【固定长度 N_ref】上再截取到各 chirp 的长度。因为 gen_burst
%            的突发数量 ~ rate·N/fs、到达时刻 ~ rand()·N/fs 都依赖长度，
%            只有固定 N_ref 才能让不同时长的 chirp 看到【逐位相同】的干扰。
%            注意：paired 模式要求 cfg.normalize = true（否则噪声幅度所依赖的
%            Pc 会随 chirp 变化，配对不成立）。
cfg.seed_mode     = lower(seed_mode);
cfg.N_ref         = 36000;                 % > 任意 chirp 的信号长度（最大约 31999）

% 突发干扰参数（老师：随机幅度、发生概率、持续时间、突发间隔）
cfg.burst = struct('rate', 5, 'dur_range', [0.002, 0.030], ...
                   'amp_range', [4, 25], 'type_prob', [0.25 0.30 0.25 0.20], ...
                   'freq_range', [200, 2600], 'bw_frac', [0.15 0.60]);

% 可选：覆盖任意突发参数（用于"干扰强度/密度"敏感性实验）。
% 用法：make_dataset_pam('alphastable', root, 1.5, sp, 'index', struct('rate',10))
if ~isempty(burst_override)
    assert(isstruct(burst_override), 'burst_override 必须是 struct');
    fn = fieldnames(burst_override);
    for ii = 1:numel(fn)
        assert(isfield(cfg.burst, fn{ii}), '未知突发参数 %s', fn{ii});
        cfg.burst.(fn{ii}) = burst_override.(fn{ii});
    end
end

if isempty(splits_override)
    splits = struct( ...
        'name',    {'train',  'val',   'testA',  'testC'}, ...
        'chirps',  {1:14,     15:17,   18:20,    18:20}, ...
        'chans',   {1:18,     19:21,   1:18,     22:25}, ...
        'n_noise', {4,        8,       4,        4});
else
    splits = splits_override;
end

fprintf('========== 生成 PAM 数据集：noise_type = %s ==========\n', cfg.noise_type);
if strcmp(cfg.noise_type, 'alphastable')
    fprintf('  α 稳定分布参数：alpha = %.2f, beta = %.2f\n', cfg.alpha, cfg.beta);
end
fprintf('  突发干扰：到达率 %.1f/s，时长 %.0f~%.0f ms，幅度 %.0f~%.0f × rms(s)\n', ...
    cfg.burst.rate, cfg.burst.dur_range(1)*1e3, cfg.burst.dur_range(2)*1e3, ...
    cfg.burst.amp_range(1), cfg.burst.amp_range(2));

%% ---- 1. 信道清单（与既有实验完全一致）--------------------------------
files = dir(fullfile(cfg.ch_dir, 'channel_H_*.mat'));
assert(~isempty(files), '在 %s 找不到 channel_H_*.mat', cfg.ch_dir);
Hmat = []; keep = [];
for k = 1:numel(files)
    Sd = load(fullfile(files(k).folder, files(k).name), 'channel_H');
    v  = Sd.channel_H(:).';
    if isempty(Hmat) || ~any(all(abs(Hmat - v) < 1e-30, 2))
        Hmat(end+1,:) = v;  keep(end+1) = k;  %#ok<AGROW>
    end
end
files = files(keep);
n_ch  = numel(files);
fprintf('  信道数 %d\n', n_ch);

%% ---- 2. chirp 参数表（与 make_dataset_gen 逐位一致）------------------
rng(cfg.seed, 'twister');
P = struct('f0',{},'f1',{},'T',{},'env',{},'phi',{},'A',{});
P(1).f0 = 798;  P(1).f1 = 1500;  P(1).T = 1.0;
P(1).env = 'rect';  P(1).phi = 0;  P(1).A = 1;
for i = 2:cfg.n_chirp_total
    a = 800  + 250*rand();
    b = 1280 + 219*rand();
    if rand() < 0.5; P(i).f0 = a;  P(i).f1 = b;
    else;            P(i).f0 = b;  P(i).f1 = a; end
    P(i).T = 0.6 + 0.4*rand();
    r = rand();
    if r < 0.5;      P(i).env = 'rect';
    elseif r < 0.75; P(i).env = 'hann';
    else;            P(i).env = 'gauss'; end
    P(i).phi = 2*pi*rand();
    P(i).A   = 0.5 + 0.5*rand();
end

%% ---- 3. 逐 split 生成 -------------------------------------------------
if ~exist(out_root, 'dir'); mkdir(out_root); end

for si = 1:numel(splits)
    sp = splits(si);
    outdir = fullfile(out_root, sp.name);
    if ~exist(outdir, 'dir'); mkdir(outdir); end
    delete(fullfile(outdir, '*.mat'));

    meta = struct('chirp_id',{},'chan_id',{},'snr',{},'file',{}, ...
                  'frac_imp',{},'peak_over_rms',{},'q75_ratio',{});
    cnt = 0; t0 = tic;

    for ci = sp.chirps
        p = P(ci);
        w = gen_chirp(cfg.fs_chirp, p.T, p.f0, p.f1, cfg.modtype, p.env, p.phi, p.A);
        w = resample(w, 5, 2);
        w = w / max(abs(w));

        for hi = sp.chans
            Sd = load(fullfile(files(hi).folder, files(hi).name), 'channel_H');
            h  = Sd.channel_H(:).';
            clean_signal = conv(w, h);
            if cfg.normalize; clean_signal = clean_signal / rms(clean_signal); end
            N = numel(clean_signal);
            Pc = mean(clean_signal.^2);
            rms_c = sqrt(Pc);

            for snr = cfg.snr_list
                Pn = Pc / 10^(snr/10);
                for rep = 1:sp.n_noise
                    cnt = cnt + 1;

                    % ---- 噪声/干扰的随机种子与生成长度 ----
                    if strcmp(cfg.seed_mode, 'paired')
                        % 与 chirp 无关：同一 (hi, snr, rep) 下所有 chirp 共享
                        % 完全相同的干扰实现，从而把波形效应单独隔离出来
                        key = (hi - 1) * 100000 + (round(snr) + 200) * 1000 + rep;
                        Ng  = cfg.N_ref;      % 固定长度 -> 突发数量/时刻与 chirp 无关
                    else
                        key = cnt;
                        Ng  = N;
                    end

                    % ---- 背景噪声 ----
                    rng(cfg.seed + key, 'twister');
                    if strcmp(cfg.noise_type, 'gauss')
                        v_full = randn(1, Ng) * sqrt(Pn);
                    else
                        x0 = stblrnd(cfg.alpha, cfg.beta, 1, 0, 1, Ng);   % 必须 1×Ng，与 clean_signal 同向
                        q0 = local_q75(abs(x0));
                        gam = sqrt(Pn) * cfg.Q75_GAUSS / max(q0, 1e-12);
                        v_full = gam * x0;
                    end
                    v = v_full(1:N);

                    % ---- 突发干扰（同一 key + 同一 Ng -> 两套数据完全一致）----
                    bc = cfg.burst;  bc.seed = cfg.seed + 500000 + key;
                    [imp_full, mask_full] = gen_burst(Ng, cfg.fs, bc);
                    imp  = imp_full(1:N) * rms_c;   % 幅度以干净信号 rms 为单位
                    mask = mask_full(1:N);

                    noisy_signal = clean_signal + v + imp;
                    imp_mask = uint8(mask(:).');

                    fimp = mean(mask);
                    pkr  = max(abs(imp)) / rms_c;
                    qr   = local_q75(abs(v)) / (sqrt(Pn) * cfg.Q75_GAUSS);

                    fname = sprintf('s_%05d.mat', cnt);
                    true_snr = snr; chirp_id = ci; chan_id = hi;
                    save(fullfile(outdir, fname), 'noisy_signal', 'clean_signal', ...
                         'imp_mask', 'imp', 'true_snr', 'chirp_id', 'chan_id');
                    meta(end+1) = struct('chirp_id', ci, 'chan_id', hi, 'snr', snr, ...
                        'file', fname, 'frac_imp', fimp, ...
                        'peak_over_rms', pkr, 'q75_ratio', qr); %#ok<AGROW>
                end
            end
        end
    end
    meta = meta(1:cnt);
    save(fullfile(out_root, sprintf('meta_%s.mat', sp.name)), 'meta', 'P', 'splits', 'cfg');
    fprintf('  [%-8s] chirps %s  chans n=%d  -> %5d 样本  %.1f min\n', ...
        sp.name, mat2str(sp.chirps), numel(sp.chans), cnt, toc(t0)/60);
end

fprintf('完成 -> %s\n', out_root);
end

% ===================== 本地工具函数 =====================
function q = local_q75(x)
x = sort(x(:));
n = numel(x);
pos = 0.75 * (n - 1) + 1;
lo = floor(pos); hi = ceil(pos);
q = x(lo) * (1 - (pos - lo)) + x(hi) * (pos - lo);
end
