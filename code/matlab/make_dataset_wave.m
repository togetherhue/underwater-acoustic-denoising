function make_dataset_wave()
%MAKE_DATASET_WAVE  波形调制泛化实验数据集：LFM 与 HFM 各生成一套
%==========================================================================
%  背景：导师 README 写明目标是「实测 HFM 信号」（双曲调频），但本项目
%        既有数据集全部用 LFM（线性调频）生成。两者只在【调频规律】上不同，
%        频带（798-1500 Hz）与时宽（0.6-1.0 s）完全一致。
%
%  本脚本用【同一张 chirp 参数表 P】分别以 LFM / HFM 生成两套数据，
%  使「调制类型」成为唯一自变量。信道划分沿用 v3 约定：
%      train   chirp 1-14  × 信道 1-18
%      val     chirp 15-17 × 信道 19-21
%      testA   chirp 18-20 × 信道 1-18     （新波形 + 见过的信道）
%      testC   chirp 18-20 × 信道 22-25    （新波形 + 新信道）
%
%  输出：
%      D:/project/dataset_wave/LFM/{train,val,testA,testC}
%      D:/project/dataset_wave/HFM/{train,val,testA,testC}
%==========================================================================

OUT = 'D:/project/dataset_wave';

splits = struct( ...
    'name',    {'train',  'val',   'testA',  'testC'}, ...
    'chirps',  {1:14,     15:17,   18:20,    18:20}, ...
    'chans',   {1:18,     19:21,   1:18,     22:25}, ...
    'n_noise', {4,        8,       4,        4});

for mod = {'LFM', 'HFM'}
    m = mod{1};
    fprintf('\n########## 生成 %s 数据集 ##########\n', m);
    make_dataset_gen(splits, fullfile(OUT, m), m);
end

fprintf('\n全部完成 -> %s\n', OUT);
end
