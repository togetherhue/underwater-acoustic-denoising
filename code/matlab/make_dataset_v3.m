function make_dataset_v3(cfg)
%MAKE_DATASET_V3  2x2 因子设计数据集：波形与信道【同时】留出
%==========================================================================
%  和 v2 的区别：v2 只留出了 chirp 波形，25 个信道全部出现在训练集里，
%  所以只验证了「新波形」泛化。本脚本把信道也分组，构造 2x2 因子设计：
%
%                    seen channels        new channels
%    seen chirps     train                testB  (只换信道)
%    new  chirps     testA (只换波形)      testC  (两者都换)
%
%  testA 用于和 v2 的结果（+13.66 dB）对照，确认训练集缩小后性能没变；
%  testB / testC 是 v2 缺失的泛化测试。
%
%  【用法】  make_dataset_v3
%==========================================================================

if nargin < 1 || isempty(cfg); cfg = struct(); end

def = struct( ...
    'ch_dir',        'D:/project/信道产生/data_fixed', ...
    'out_root',      'D:/project/dataset_v3', ...
    'fs_chirp',      9600, ...
    'fs_ch',         24000, ...
    'n_chirp_total', 20, ...
    'train_chirps',  1:14, ...
    'val_chirps',    15:17, ...
    'test_chirps',   18:20, ...
    'train_chans',   1:18, ...
    'val_chans',     19:21, ...
    'test_chans',    22:25, ...
    'n_noise_train', 6, ...
    'n_noise_val',   8, ...
    'n_noise_test',  4, ...
    'n_noise_testC', 8, ...
    'snr_list',      [-10 -5 0 5 10], ...
    'modtype',       'LFM', ...
    'normalize',     true, ...
    'seed',          20261013);
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(cfg, fn{k}); cfg.(fn{k}) = def.(fn{k}); end
end
rng(cfg.seed, 'twister');

%% ---- 1. 信道清单（与 v2 完全一致：MATLAB dir 顺序 + 内容去重）----------
files = dir(fullfile(cfg.ch_dir, 'channel_H_*.mat'));
assert(~isempty(files), '在 %s 找不到 channel_H_*.mat', cfg.ch_dir);
Hmat = []; keep = [];
for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name), 'channel_H');
    v  = S.channel_H(:).';
    if isempty(Hmat) || ~any(all(abs(Hmat - v) < 1e-30, 2))
        Hmat(end+1,:) = v;  keep(end+1) = k;  %#ok<AGROW>
    end
end
if numel(keep) < numel(files)
    fprintf('信道去重：%d -> %d\n', numel(files), numel(keep));
end
files = files(keep);
n_ch  = numel(files);
fprintf('信道总数：%d\n', n_ch);

assert(max([cfg.train_chans cfg.val_chans cfg.test_chans]) <= n_ch, ...
    '信道编号超出范围（只有 %d 个）', n_ch);
assert(isempty(intersect(cfg.train_chans, cfg.test_chans)) && ...
       isempty(intersect(cfg.train_chans, cfg.val_chans)) && ...
       isempty(intersect(cfg.val_chans,   cfg.test_chans)), '信道分组有重叠');
assert(isempty(intersect(cfg.train_chirps, cfg.test_chirps)) && ...
       isempty(intersect(cfg.train_chirps, cfg.val_chirps)) && ...
       isempty(intersect(cfg.val_chirps,   cfg.test_chirps)), 'chirp 分组有重叠');

%% ---- 2. chirp 参数表（与 v2 逐位一致，保证 testA 可与 v2 对照）---------
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
    if r < 0.5;     P(i).env = 'rect';
    elseif r < 0.75; P(i).env = 'hann';
    else;            P(i).env = 'gauss'; end
    P(i).phi = 2*pi*rand();
    P(i).A   = 0.5 + 0.5*rand();
end
fprintf('chirp：train %d 组 / val %d 组 / test %d 组\n', ...
    numel(cfg.train_chirps), numel(cfg.val_chirps), numel(cfg.test_chirps));
fprintf('信道：train %d 个 / val %d 个 / test %d 个\n', ...
    numel(cfg.train_chans), numel(cfg.val_chans), numel(cfg.test_chans));

%% ---- 3. 五个 split ------------------------------------------------------
SPL(1) = mk('train',                cfg.train_chirps, cfg.train_chans, cfg.n_noise_train);
SPL(2) = mk('val',                  cfg.val_chirps,   cfg.val_chans,   cfg.n_noise_val);
SPL(3) = mk('testA_seench_newchirp',cfg.test_chirps,  cfg.train_chans, cfg.n_noise_test);
SPL(4) = mk('testB_newch_seenchirp',cfg.train_chirps, cfg.test_chans,  cfg.n_noise_test);
SPL(5) = mk('testC_newch_newchirp', cfg.test_chirps,  cfg.test_chans,  cfg.n_noise_testC);

fprintf('\n%-24s %-10s %-10s %8s %8s\n', 'split', 'chirps', 'chans', '样本数', '磁盘GB');
tot = 0; totgb = 0;
for s = SPL
    L = round(P(s.chirps(1)).T*cfg.fs_chirp)*(cfg.fs_ch/cfg.fs_chirp) + 8000 - 1;
    gb = s.n * L * 8 * 2 / 1e9;          % double 存储，noisy+clean 两个数组
    fprintf('%-24s %-10s %-10s %8d %8.2f\n', s.name, ...
        sprintf('%d-%d', s.chirps(1), s.chirps(end)), ...
        sprintf('%d-%d', s.chans(1), s.chans(end)), s.n, gb);
    tot = tot + s.n; totgb = totgb + gb;
end
fprintf('%-24s %-10s %-10s %8d %8.2f\n\n', '合计', '', '', tot, totgb);

%% ---- 4. 生成 ------------------------------------------------------------
for s = SPL
    outdir = fullfile(cfg.out_root, s.name);
    if ~exist(outdir,'dir'); mkdir(outdir); end
    delete(fullfile(outdir,'*.mat'));
    meta = struct('chirp_id',{},'chan_id',{},'snr',{},'file',{});
    cnt = 0; t0 = tic;
    for ci = s.chirps
        p = P(ci);
        w = gen_chirp(cfg.fs_chirp, p.T, p.f0, p.f1, cfg.modtype, p.env, p.phi, p.A);
        w = resample(w, 5, 2);
        w = w / max(abs(w));
        for hi = s.chans
            Sd = load(fullfile(files(hi).folder, files(hi).name), 'channel_H');
            h  = Sd.channel_H(:).';
            clean_signal = conv(w, h);
            if cfg.normalize; clean_signal = clean_signal / rms(clean_signal); end
            Pc = mean(clean_signal.^2);
            for snr = cfg.snr_list
                Pn = Pc / 10^(snr/10);
                for rep = 1:s.n_noise
                    noisy_signal = clean_signal + randn(size(clean_signal)) * sqrt(Pn);
                    cnt = cnt + 1;
                    fname = sprintf('s_%05d.mat', cnt);
                    true_snr = snr; chirp_id = ci; chan_id = hi;
                    save(fullfile(outdir, fname), ...
                         'noisy_signal','clean_signal','true_snr','chirp_id','chan_id');
                    meta(end+1) = struct('chirp_id',ci,'chan_id',hi,'snr',snr, ...
                        'file',fname); %#ok<AGROW>
                end
            end
        end
    end
    meta = meta(1:cnt);
    save(fullfile(cfg.out_root, sprintf('meta_%s.mat', s.name)), 'meta', 'P', 'cfg');
    fprintf('  [%-22s] %5d 样本  %.1f min\n', s.name, cnt, toc(t0)/60);
end
save(fullfile(cfg.out_root, 'split_info.mat'), 'SPL', 'P', 'cfg');
fprintf('\n全部完成 -> %s\n', cfg.out_root);
end


function s = mk(name, chirps, chans, n_noise)
s = struct('name', name, 'chirps', chirps, 'chans', chans, ...
           'n_noise', n_noise, ...
           'n', numel(chirps)*numel(chans)*5*n_noise);
end
