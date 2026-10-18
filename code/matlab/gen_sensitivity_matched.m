function gen_sensitivity_matched(idx)
%GEN_SENSITIVITY_MATCHED  等有效功率 SNR 的 α 重扫
%==========================================================================
% 目的：补齐"适用边界"实验里唯一的方法论缺口 —— 把 α 与噪声强度解耦。
%
% 问题：名义 SNR 用 75% 分位数匹配定义（α<2 时方差无穷，无法用功率定义）。
% 这固定了噪声"核心宽度"，但没固定二阶矩。实测 shift = 有效功率SNR − 名义SNR：
%       α=1.5 : +6.71 dB（base）
%       α=1.2 : −6.36 dB   → 同一名义 SNR 下，α=1.2 的有效噪声功率高 13.1 dB
%       α=1.8 : +16.71 dB  → 低 10.0 dB
%       α=2.0 : +20.84 dB  → 低 14.1 dB
% （shift 用中位数估计，比均值稳健 —— α 小时样本方差本身是重尾量）
%
% 因此 gen_sensitivity.m 的 α 扫描实际上同时变了两个因素：尾形 + 强度。
% 本脚本把名义 SNR 平移对应的量，使【有效功率 SNR 与 base 一一对应】，
% 从而把 α 的效应单独隔离出来。
%
% 关键：平移后样本序号 i 仍与 base 对应同一 (chirp, 信道, SNR档, 重复)，
% 且底层均匀随机数（同 seed）与突发实现（同 seed + 同 rms_c）完全一致 ——
% 只有背景噪声的尾形不同。构成干净的逐样本配对。
%
% 条件：
%   m_a12   α=1.2, 名义 SNR = base + 13.07 dB
%   m_a18   α=1.8, 名义 SNR = base − 10.00 dB
%
% 用法：gen_sensitivity_matched(1) / gen_sensitivity_matched(2) / (无参 = 全部)
%==========================================================================

BASE = [-10 -5 0 5 10];
CONDS = {
    'm_a12',  1.2,  BASE + 13.07;
    'm_a18',  1.8,  BASE - 10.00;
};

if nargin < 1 || isempty(idx); idx = 1:size(CONDS,1); end
out_base = 'E:/project/data/ds_sens_matched';

for k = idx
    name = CONDS{k,1};
    al   = CONDS{k,2};
    sl   = CONDS{k,3};
    sp = struct( ...
        'name',    {'train',  'val',   'testA',  'testC'}, ...
        'chirps',  {1:14,     15:17,   18:20,    18:20}, ...
        'chans',   {1:18,     19:21,   1:18,     22:25}, ...
        'n_noise', {4,        8,       4,        4});
    fprintf('\n############ 等功率匹配条件 %s : alpha=%.2f  snr_list=[%s] ############\n', ...
        name, al, num2str(sl, ' %+0.2f'));
    make_dataset_pam('alphastable', fullfile(out_base, name), al, sp, 'index', [], sl);
end
fprintf('\n全部完成 -> %s\n', out_base);
end
