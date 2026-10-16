function gen_sensitivity(idx)
%GEN_SENSITIVITY  生成"适用边界"敏感性实验的数据集
%==========================================================================
% 目的：当前全部结论都建立在【α=1.5、突发到达率 5/s】这一个工作点上。
% 本脚本用【单因素轮换】设计，围绕该工作点各变一个方向，检验结论的适用边界。
%
%   条件         α      rate(1/s)   检验的假设
%   ---------------------------------------------------------------
%   a12_r5      1.2      5         背景噪声更脉冲化 -> 脉冲感知分支是否更有价值？
%   a18_r5      1.8      5         背景噪声更接近高斯
%   a20_r5      2.0      5         α=2 应退化为高斯（与 gauss 数据集互为校验）
%   a15_r25     1.5      2.5       干扰更稀疏 -> 频域掩码应更容易处理
%   a15_r10     1.5     10         干扰更密集 -> 频域掩码应更难处理
%   (基线)      1.5      5         既有 dataset_pam/alphastable
%
% 生成【完整四个 split】，使得既可以做零样本跨条件评估（用 testA/testC），
% 也可以在异常条件上做域内重训（用 train/val），从而区分
%   「分布偏移造成的退化」 与 「该条件下方法本身的优劣」。
%
% 用法：gen_sensitivity(1) ... gen_sensitivity(5)
%==========================================================================

CONDS = {
    'a12_r5',  1.2,  5.0,   struct();
    'a18_r5',  1.8,  5.0,   struct();
    'a20_r5',  2.0,  5.0,   struct();
    'a15_r25', 1.5,  2.5,   struct();
    'a15_r10', 1.5, 10.0,   struct();
};

if nargin < 1 || isempty(idx); idx = 1:size(CONDS,1); end

out_base = 'D:/project/ds_sens';

% 只生成 testA / testC（评估用）时把下面注释打开可大幅提速；
% 但域内重训需要 train/val，因此默认生成完整 split。
full = true;

for k = idx
    name = CONDS{k,1};
    al   = CONDS{k,2};
    rate = CONDS{k,3};
    bo   = CONDS{k,4};
    bo.rate = rate;

    if full
        sp = struct( ...
            'name',    {'train',  'val',   'testA',  'testC'}, ...
            'chirps',  {1:14,     15:17,   18:20,    18:20}, ...
            'chans',   {1:18,     19:21,   1:18,     22:25}, ...
            'n_noise', {4,        8,       4,        4});
    else
        sp = struct( ...
            'name',    {'testA',  'testC'}, ...
            'chirps',  {18:20,    18:20}, ...
            'chans',   {1:18,     22:25}, ...
            'n_noise', {4,        4});
    end

    fprintf('\n############ 条件 %s : alpha=%.2f  rate=%.1f/s ############\n', ...
        name, al, rate);
    make_dataset_pam('alphastable', fullfile(out_base, name), al, sp, 'index', bo);
end
fprintf('\n全部完成 -> %s\n', out_base);
end
