%% =========================================================================
%  fix_make_dataset.m   ——  修正版「训练 / 测试 / 微调 数据集」生成脚本
% -------------------------------------------------------------------------
%  【作用】
%      读入干净 chirp + data_fixed/ 里的信道冲激响应，
%      生成三套 .mat 数据集：
%          D:/project/tr_signal_new/        训练集
%          D:/project/te_signal_new/        测试集
%          D:/project/finetune_signal_new/  微调集
%      每个样本文件里只有一个变量 noisy_signal，外加 true_snr、chan_idx。
%      干净目标单独存一份：D:/project/chirp_signal_24k.mat（变量 clean_signal）
%
%  【修的是什么】
%      chirp_signal.mat 是 9600 Hz 采样的（实测 99% 能量在归一化频率
%      0.08337~0.15607，×9600 = 800~1498 Hz，正好是换能器频带 798~1500 Hz），
%      而信道是在 24000 Hz 上定义的（Len=8000, df=3Hz）。
%      两者必须先统一采样率，否则频带完全不重叠，conv(chirp,H) ≈ 0。
%      所以本脚本先 resample(chirp, 24000, 9600) 再卷积。
%
%  【怎么用】
%      1) 先跑完 fix_channel_gen.m（生成 data_fixed/channel_H_*.mat）
%      2) MATLAB 里直接运行本脚本（不需要 cd，路径都写绝对路径了）
%      3) 跑完后按屏幕提示自检
%
%  【注意】重采样后 chirp 长 24000 点，卷积后信号长 31999 点，
%          比原来的 18549 长。用新数据训练时要把 Python 里的
%          input_size 从 18549 改成 31999（或按需要裁剪）。
% =========================================================================
clc; clear; close all;

%% ---- 1. 配置 -------------------------------------------------------------
fs_chirp   = 9600;      % chirp 文件的采样率（实测反推）
fs_channel = 24000;     % 信道的采样率

N_train = 6000;         % 训练样本数
N_test  = 11;           % 测试样本数
N_fine  = 13;           % 微调样本数

snr_list = [-10 -5 0 5 10];   % 目标真实 SNR [dB]，按此随机分层

ch_dir     = 'D:/project/信道产生/data_fixed';
chirp_file = 'D:/project/chirp_signal.mat';

out_tr      = 'D:/project/tr_signal_new';
out_te      = 'D:/project/te_signal_new';
out_fn      = 'D:/project/finetune_signal_new';
target_file = 'D:/project/chirp_signal_24k.mat';

for d = {out_tr, out_te, out_fn}
    if ~exist(d{1}, 'dir'); mkdir(d{1}); end
end

%% ---- 2. 读 chirp 并重采样到 24000 Hz -------------------------------------
S  = load(chirp_file);
fn = fieldnames(S);
fn = fn(~strncmp(fn, '__', 2));
clean = S.(fn{1})(:).';

clean24 = resample(clean, 5, 2);        % 9600 → 24000（比值 2.5 = 5/2）
clean24 = clean24 / max(abs(clean24));  % 归一化到 ±1

fprintf('chirp: %d 点 @%d Hz (%.3f s)  →  %d 点 @%d Hz (%.3f s)\n', ...
    numel(clean),  fs_chirp,   numel(clean)/fs_chirp, ...
    numel(clean24), fs_channel, numel(clean24)/fs_channel);

% 自检：重采样后频带应落在 0.0333~0.0625
N = 2^nextpow2(numel(clean24));
P = abs(fft(clean24, N)).^2;  P = P(1:N/2+1);
E = cumsum(P)/sum(P);
lo = find(E >= 0.005, 1)/N;
hi = find(E >= 0.995, 1)/N;
fprintf('重采样后 chirp 归一化频带 = %.5f ~ %.5f  （目标 0.0333 ~ 0.0625）\n', lo, hi);
if lo > 0.025 && hi < 0.075
    fprintf('✓ 频带对齐正确\n\n');
else
    error('频带仍然不对，请检查 fs_chirp / fs_channel');
end

clean_signal = clean24;
save(target_file, 'clean_signal', 'fs_channel');
fprintf('干净目标已保存 → %s\n\n', target_file);

%% ---- 3. 信道清单 ---------------------------------------------------------
files = dir(fullfile(ch_dir, 'channel_H_*.mat'));
assert(~isempty(files), ...
    '在 %s 里找不到 channel_H_*.mat，请先运行 fix_channel_gen.m', ch_dir);
fprintf('可用信道：%d 个\n\n', numel(files));

%% ---- 4. 生成三套数据 -----------------------------------------------------
make_set(out_tr, 'tr', N_train, files, clean24, snr_list);
make_set(out_te, 'te', N_test,  files, clean24, snr_list);
make_set(out_fn, 'ft', N_fine,  files, clean24, snr_list);

fprintf('\n============================ 完成 ============================\n');
fprintf('训练集 → %s\n测试集 → %s\n微调集 → %s\n', out_tr, out_te, out_fn);
fprintf('干净目标 → %s（变量名 clean_signal）\n', target_file);
fprintf('\n自检方法：\n');
fprintf('  1) 含噪信号的 99%% 能量应落在归一化频率 0.0333~0.0625\n');
fprintf('  2) 与 clean_signal 的互相关峰应在 0 延迟附近，且波形展宽（多径）\n');
fprintf('  3) 每个文件里的 true_snr 就是该样本的真实 SNR，可直接当标签用\n');
fprintf('==============================================================\n');


%% =========================================================================
%  局部函数
% =========================================================================
function make_set(outdir, tag, N, files, clean24, snr_list)
    for i = 1:N
        % --- 随机抽一个信道 ---
        f = files(randi(numel(files)));
        S = load(fullfile(f.folder, f.name), 'channel_H');
        h = S.channel_H(:).';

        % --- 信号经过信道（保留物理幅度，不做逐样本归一化）---
        y = conv(clean24, h);

        % --- 按目标 SNR 加高斯噪声 ---
        snr = snr_list(randi(numel(snr_list)));
        Py  = mean(y.^2);
        Pn  = Py / 10^(snr/10);
        n   = randn(size(y)) * sqrt(Pn);

        noisy_signal = y + n;
        true_snr     = snr;
        chan_idx     = f.name;

        save(fullfile(outdir, sprintf('%s_%04d.mat', tag, i)), ...
             'noisy_signal', 'true_snr', 'chan_idx');
    end
    fprintf('已生成 %4d 个样本 → %s\n', N, outdir);
end
