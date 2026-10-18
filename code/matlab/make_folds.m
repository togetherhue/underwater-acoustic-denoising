function make_folds()
%MAKE_FOLDS  按时延扩展【分层】做 3 折信道留出，生成 3 套数据集
%==========================================================================
%  目的：把「信道泛化代价」这个结论从"只测了 4 个信道"提升为
%        "3 个独立折 × 每折 8-9 个信道"。
%
%  分层方式：把 25 个信道按时延扩展排序，按 rank 轮转分到 3 折。
%            这样每折都横跨整个难度范围（从近单径到强多径），
%            比较「留出信道」与「见过的信道」时难度分布天然匹配。
%
%  每折产出 4 个 split：
%      train        chirp 1-14  × 训练信道      （模型学到的）
%      val          chirp 15-17 × 验证信道      （波形与信道都新）
%      test_seen    chirp 18-20 × 训练信道      ← 见过的信道，新波形
%      test_unseen  chirp 18-20 × 留出信道      ← 没见过的信道，新波形
%  test_seen 与 test_unseen 用同一批波形，且信道难度分布匹配
%  => 两者之差就是纯粹的「信道新颖度代价」
%==========================================================================

CH  = 'E:/project/data/信道产生/data_fixed';
OUT = 'D:/project/dataset_folds';
FSch = 24000;
N_FOLD = 3;

%% ---- 1. 信道清单（与 make_dataset_gen 完全一致）------------------------
files = dir(fullfile(CH, 'channel_H_*.mat'));
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
fprintf('信道总数 %d\n', n_ch);

%% ---- 2. 时延扩展 + 分层折 ----------------------------------------------
sp = zeros(1, n_ch);
for k = 1:n_ch
    h = Hmat(k,:);
    e = abs(h).^2;  e = e/sum(e);  c = cumsum(e);
    sp(k) = (find(c>=0.95,1) - find(c>=0.05,1)) / FSch * 1000;
end
[~, o] = sort(sp);
fold = zeros(1, n_ch);
for r = 1:n_ch
    fold(o(r)) = mod(r-1, N_FOLD) + 1;
end

fprintf('\n=== 分层结果（chan_id 为 MATLAB dir 字典序下标）===\n');
for k = 1:N_FOLD
    idx = find(fold == k);
    v = sp(idx);
    fprintf('  fold %d: n=%d  时延 %.1f~%.1f ms  中位 %.1f   chan_id %s\n', ...
        k, numel(idx), min(v), max(v), median(v), mat2str(idx));
end

%% ---- 3. 逐折生成 --------------------------------------------------------
for k = 1:N_FOLD
    held = find(fold == k);
    pool = find(fold ~= k);
    valc = pool(1:2);            % 2 个信道做验证（波形与信道都是新的）
    trc  = pool(3:end);          % 其余做训练

    splits = struct( ...
        'name',    {'train',       'val',    'test_seen',  'test_unseen'}, ...
        'chirps',  {1:14,          15:17,    18:20,        18:20}, ...
        'chans',   {trc,           valc,     trc,          held}, ...
        'n_noise', {4,             8,        4,            4});

    fprintf('\n===== fold %d =====\n', k);
    fprintf('  留出(测试)信道 %d 个: %s\n', numel(held), mat2str(held));
    fprintf('  训练信道 %d 个, 验证信道 %d 个: %s\n', numel(trc), numel(valc), mat2str(valc));
    make_dataset_gen(splits, fullfile(OUT, sprintf('fold%d', k)));
end

save(fullfile(OUT, 'folds.mat'), 'fold', 'sp', 'files', 'n_ch');
fprintf('\n折划分已保存 -> %s/folds.mat\n', OUT);
end
