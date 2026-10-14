function make_dataset_gen(splits, out_root)
%MAKE_DATASET_GEN  通用数据集生成器：按任意 split 定义生成样本
%==========================================================================
%  与 make_dataset_v3 的区别：split 由调用方任意指定，不做固定 2x2 布局。
%
%  splits : 结构体数组，每个元素含
%             .name     目录名
%             .chirps   chirp 编号列表（1..20）
%             .chans    信道编号列表（1..n_ch，按 MATLAB dir 字典序）
%             .n_noise  该 split 的噪声实现次数
%  out_root : 输出根目录
%
%  为保证 chirp 波形与既有实验【逐位一致】，P 表的生成方式与
%  make_dataset_v3 完全相同（同 seed + 同 rand 调用顺序）。
%==========================================================================

cfg.fs_chirp     = 9600;
cfg.fs_ch         = 24000;
cfg.n_chirp_total = 20;
cfg.snr_list      = [-10 -5 0 5 10];
cfg.modtype       = 'LFM';
cfg.normalize     = true;
cfg.seed          = 20261013;
cfg.ch_dir        = 'D:/project/信道产生/data_fixed';

rng(cfg.seed, 'twister');

%% ---- 1. 信道清单（与既有实验一致：MATLAB dir 顺序 + 内容去重）----------
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

%% ---- 2. chirp 参数表（与 make_dataset_v3 逐位一致）---------------------
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

%% ---- 3. 逐个 split 生成 -------------------------------------------------
for si = 1:numel(splits)
    sp = splits(si);
    assert(max(sp.chans) <= n_ch, 'split %s 的信道编号超出范围', sp.name);
    outdir = fullfile(out_root, sp.name);
    if ~exist(outdir,'dir'); mkdir(outdir); end
    delete(fullfile(outdir,'*.mat'));

    meta = struct('chirp_id',{},'chan_id',{},'snr',{},'file',{});
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
            Pc = mean(clean_signal.^2);
            for snr = cfg.snr_list
                Pn = Pc / 10^(snr/10);
                for rep = 1:sp.n_noise
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
    save(fullfile(out_root, sprintf('meta_%s.mat', sp.name)), 'meta', 'P', 'splits');
    fprintf('  [%-14s] chirps %s  chans n=%d  ->  %5d 样本  %.1f min\n', ...
        sp.name, mat2str(sp.chirps), numel(sp.chans), cnt, toc(t0)/60);
end
fprintf('完成 -> %s\n', out_root);
end
