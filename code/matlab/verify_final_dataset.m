%% 数据集最终验证
clc; clear; rng(7);
root = 'D:/project/dataset';

%% 1. 数量
fprintf('=== 样本数 ===\n');
for s = {'train','val','test'}
    d = dir(fullfile(root, s{1}, '*.mat'));
    fprintf('  %-6s %5d 个\n', s{1}, numel(d));
end

%% 2. 按 chirp_id / chan_id 分布
fprintf('\n=== 分布 ===\n');
for s = {'train','val','test'}
    d = dir(fullfile(root, s{1}, '*.mat'));
    cid = zeros(1,numel(d)); hid = zeros(1,numel(d)); sn = zeros(1,numel(d));
    for k = 1:numel(d)
        S = load(fullfile(d(k).folder,d(k).name),'chirp_id','chan_id','true_snr');
        cid(k)=S.chirp_id; hid(k)=S.chan_id; sn(k)=S.true_snr;
    end
    fprintf('  %-6s chirp_id=%s  信道数=%d  SNR=%s\n', s{1}, ...
        mat2str(unique(cid)), numel(unique(hid)), mat2str(unique(sn)));
end

%% 3. 划分是否真的按 chirp 分组（关键！）
fprintf('\n=== 划分独立性（关键）===\n');
g = @(s) getid(root, s);
tr = g('train'); va = g('val'); te = g('test');
fprintf('  train chirp: %s\n', mat2str(tr));
fprintf('  val   chirp: %s\n', mat2str(va));
fprintf('  test  chirp: %s\n', mat2str(te));
fprintf('  train ∩ test = %s   %s\n', mat2str(intersect(tr,te)), ...
    ternary(isempty(intersect(tr,te)), '✓ 无重叠', '✗ 有重叠！'));
fprintf('  train ∩ val  = %s   %s\n', mat2str(intersect(tr,va)), ...
    ternary(isempty(intersect(tr,va)), '✓ 无重叠', '✗ 有重叠！'));

%% 4. 干净目标多样性（最关键）
fprintf('\n=== 干净目标多样性 ===\n');
for s = {'train','test'}
    d = dir(fullfile(root, s{1}, '*.mat'));
    idx = randperm(numel(d), 12);
    C = {};
    for k = idx
        S = load(fullfile(d(k).folder,d(k).name),'clean_signal');
        C{end+1} = S.clean_signal(:);
    end
    cc = [];
    for i = 1:numel(C)
        for j = i+1:numel(C)
            m = min(numel(C{i}),numel(C{j}));
            v = corrcoef(C{i}(1:m), C{j}(1:m)); cc(end+1)=v(1,2);
        end
    end
    fprintf('  %-6s 12 个样本 clean 两两相关: min=%.4f 平均=%.4f max=%.4f\n', ...
        s{1}, min(cc), mean(cc), max(cc));
end
fprintf('  （旧数据所有样本目标相同 → 恒为 1.0000）\n');

%% 5. SNR 精度
fprintf('\n=== SNR 精度 ===\n');
d = dir(fullfile(root,'train','*.mat'));
idx = randperm(numel(d), 10);
for k = idx
    S = load(fullfile(d(k).folder,d(k).name));
    meas = 10*log10( sum(S.clean_signal.^2)/sum((S.noisy_signal-S.clean_signal).^2) );
    fprintf('  目标 %+3g dB -> 实测 %+7.3f dB   长度 %d\n', S.true_snr, meas, numel(S.noisy_signal));
end

%% 6. 频带
fprintf('\n=== 频带 ===\n');
for s = {'train','test'}
    d = dir(fullfile(root, s{1}, '*.mat'));
    S = load(fullfile(d(1).folder,d(1).name));
    for fld = {'clean_signal','noisy_signal'}
        v = S.(fld{1})(:).'; N = 2^nextpow2(numel(v));
        P = abs(fft(v,N)).^2; P=P(1:N/2+1); E=cumsum(P)/sum(P);
        fprintf('  %-6s %-14s 99%%频带 = %.5f ~ %.5f\n', s{1}, fld{1}, ...
            find(E>=0.005,1)/N, find(E>=0.995,1)/N);
    end
end

%% 7. 参数量
fprintf('\n=== 多样性总量 ===\n');
P = load(fullfile(root,'chirp_params.mat'));
nch = 25;
fprintf('  chirp 参数组：%d\n', numel(P.P));
fprintf('  信道：%d\n', nch);
fprintf('  不同干净目标数（chirp × 信道）= %d\n', numel(P.P)*nch);
fprintf('  总样本：10000\n');


function ids = getid(root, s)
d = dir(fullfile(root, s, '*.mat'));
ids = [];
for k = 1:numel(d)
    S = load(fullfile(d(k).folder,d(k).name),'chirp_id');
    ids(end+1) = S.chirp_id;
end
ids = unique(ids);
end

function out = ternary(c, a, b)
if c; out = a; else; out = b; end
end
