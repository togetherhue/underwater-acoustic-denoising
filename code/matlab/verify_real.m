%% 验证真实生成的信道（fct=10）是否正确、是否物理合理
clc; clear;
fs=24000; Len=8000; df=fs/Len;

files = dir('E:/project/data/信道产生/data_fixed/channel_H_*.mat');
fprintf('共 %d 个信道\n\n', numel(files));
fprintf('%-20s %-22s %-12s %-12s %s\n', '文件','99%频带','峰值位置','max|h|','时延扩展90%');
fprintf('%s\n', repmat('-',1,84));

npass = 0;
for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name), 'channel_H');
    h = S.channel_H(:).';
    P = abs(fft(h(:), Len)).^2; P = P(1:Len/2+1);
    E = cumsum(P)/sum(P);
    lo = find(E>=0.005,1)/Len;
    hi = find(E>=0.995,1)/Len;
    pk = find(abs(h)==max(abs(h)),1);
    e  = cumsum(h.^2)/sum(h.^2);
    d90 = find(e>=0.9,1) - pk;

    ok = (lo>0.025 && hi<0.075);
    npass = npass + ok;
    fprintf('%-20s %.5f~%.5f  %-12d %-12.3e %d 点 (%.1f ms)  %s\n', ...
        files(k).name, lo, hi, pk, max(abs(h)), abs(d90), abs(d90)/fs*1000, ...
        string(ok));
end
fprintf('\n频带正确: %d / %d\n', npass, numel(files));

%% 与旧信道对比
fprintf('\n============ 与旧信道对比 ============\n');
oldf = dir('E:/project/data/信道产生/data/channel_H_1.mat');
S1 = load(fullfile(oldf(1).folder, oldf(1).name), 'channel_H');
h1 = S1.channel_H(:).';
P1 = abs(fft(h1(:), Len)).^2; P1 = P1(1:Len/2+1);
E1 = cumsum(P1)/sum(P1);
fprintf('旧信道 99%%频带 = %.5f ~ %.5f   (错误，应为 0.0333~0.0625)\n', ...
    find(E1>=0.005,1)/Len, find(E1>=0.995,1)/Len);

S2 = load('E:/project/data/信道产生/data_fixed/channel_H_1.mat', 'channel_H');
h2 = S2.channel_H(:).';
fprintf('新信道 99%%频带 = %.5f ~ %.5f   (正确)\n', ...
    find(cumsum(abs(fft(h2(:),Len)).^2)./sum(abs(fft(h2(:),Len)).^2)>=0.005,1)/Len, ...
    find(cumsum(abs(fft(h2(:),Len)).^2)./sum(abs(fft(h2(:),Len)).^2)>=0.995,1)/Len);

%% 端到端：用真实新信道跑一遍
fprintf('\n============ 端到端（用真实新信道）============\n');
S = load('D:/project/chirp_signal.mat');
fn = fieldnames(S); fn = fn(~strncmp(fn,'__',2));
clean = S.(fn{1})(:).';
clean24 = resample(clean, 5, 2); clean24 = clean24/max(abs(clean24));

for k = [1 5 9]
    Sk = load(sprintf('E:/project/data/信道产生/data_fixed/channel_H_%d.mat', k), 'channel_H');
    hk = Sk.channel_H(:).';
    y = conv(clean24, hk);
    N = 2^nextpow2(numel(y));
    Py = abs(fft(y,N)).^2; Py = Py(1:N/2+1);
    Ey = cumsum(Py)/sum(Py);
    lo = find(Ey>=0.005,1)/N; hi = find(Ey>=0.995,1)/N;
    c = real(ifft(fft(y,N).*conj(fft([clean24, zeros(1,N-numel(clean24))]))));
    [cm,ci] = max(abs(c(1:numel(y))));
    m = true(1,numel(y)); m(max(1,ci-400):min(numel(y),ci+400)) = false;
    fprintf('ch%-3d: 卷积后频带 %.5f~%.5f  匹配滤波峰@%d (信道主到达@%d) 峰/底噪=%.0f倍(%.1f dB)\n', ...
        k, lo, hi, ci, find(abs(hk)==max(abs(hk)),1), cm/rms(c(m)), 20*log10(cm/rms(c(m))));
end
