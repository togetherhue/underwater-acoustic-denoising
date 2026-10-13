%% 验证生成的数据集
clc; clear;
root = 'D:/project/dataset_test';
D = dir(fullfile(root,'train','*.mat'));
fprintf('train 样本数: %d\n\n', numel(D));

%% 1. 长度与 SNR 准确性
fprintf('%-14s %-8s %-10s %-10s %-10s %s\n','文件','长度','目标SNR','实测SNR','chirp_id','chan_id');
n = min(numel(D), 6);
for k = 1:n
    S = load(fullfile(D(k).folder, D(k).name));
    L = numel(S.noisy_signal);
    meas = 10*log10( sum(S.clean_signal.^2) / sum((S.noisy_signal-S.clean_signal).^2) );
    fprintf('%-14s %-8d %-10g %-10.3f %-10d %d\n', ...
        D(k).name, L, S.true_snr, meas, S.chirp_id, S.chan_id);
end

%% 2. 关键：不同信道的 clean 目标是否真的不同
fprintf('\n=== 不同信道 → 干净目标是否不同 ===\n');
C = {};
for k = 1:min(numel(D), 8)
    S = load(fullfile(D(k).folder, D(k).name), 'clean_signal');
    C{end+1} = S.clean_signal(:);
end
cc = [];
for i = 1:numel(C)
    for j = i+1:numel(C)
        a = C{i}; b = C{j};
        m = min(numel(a),numel(b)); a=a(1:m); b=b(1:m);
        ctmp = corrcoef(a,b);
        cc(end+1) = ctmp(1,2);
    end
end
fprintf('  %d 个样本的 clean 两两相关系数：min=%.4f  平均=%.4f  max=%.4f\n', ...
        numel(C), min(cc), mean(cc), max(cc));
fprintf('  （旧数据里所有样本目标相同 → 相关系数恒为 1.0000）\n');

%% 3. 频带检查
fprintf('\n=== 频带检查 ===\n');
for k = 1:min(numel(D),3)
    S = load(fullfile(D(k).folder, D(k).name));
    for fld = {'noisy_signal','clean_signal'}
        v = S.(fld{1})(:).';
        N = 2^nextpow2(numel(v));
        P = abs(fft(v,N)).^2; P = P(1:N/2+1);
        E = cumsum(P)/sum(P);
        fprintf('  %s %-14s 99%%频带 = %.5f ~ %.5f\n', D(k).name, fld{1}, ...
            find(E>=0.005,1)/N, find(E>=0.995,1)/N);
    end
end

%% 4. 不同 SNR 档位的效果（如果有）
fprintf('\n=== 各 SNR 档位样本数 ===\n');
allD = dir(fullfile(root,'train','*.mat'));
snrs = [];
for k = 1:numel(allD)
    S = load(fullfile(allD(k).folder, allD(k).name), 'true_snr');
    snrs(end+1) = S.true_snr;
end
u = unique(snrs);
for k = 1:numel(u)
    fprintf('  SNR = %4g dB : %d 个\n', u(k), sum(snrs==u(k)));
end

%% 5. 与旧数据对比
fprintf('\n=== 与旧数据对比 ===\n');
old = load('D:/project/te_signal/test_sample_1.mat','noisy_signal');
o = old.noisy_signal(:).';
N = 2^nextpow2(numel(o)); P = abs(fft(o,N)).^2; P=P(1:N/2+1);
E = cumsum(P)/sum(P);
fprintf('  旧含噪信号 99%%频带 = %.5f ~ %.5f   （错误，应为 0.0333~0.0625）\n', ...
        find(E>=0.005,1)/N, find(E>=0.995,1)/N);
