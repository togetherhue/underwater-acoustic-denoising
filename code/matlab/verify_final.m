%% 端到端验证：chirp 能不能通过修正后的信道
clc; clear;

%% 1. 干净 chirp（9600 Hz）重采样到 24000 Hz
S = load('D:/project/chirp_signal.mat');
fn = fieldnames(S); fn = fn(~strncmp(fn,'__',2));
clean = S.(fn{1})(:).';
clean24 = resample(clean, 5, 2);
clean24 = clean24 / max(abs(clean24));
fprintf('chirp: %d 点 @9600Hz  ->  %d 点 @24000Hz\n', numel(clean), numel(clean24));

N1 = 2^nextpow2(numel(clean24));
P1 = abs(fft(clean24,N1)).^2; P1 = P1(1:N1/2+1);
E1 = cumsum(P1)/sum(P1);
fprintf('重采样后 chirp 频带 = %.5f ~ %.5f   （信道带 0.0333~0.0625）\n\n', ...
        find(E1>=0.005,1)/N1, find(E1>=0.995,1)/N1);

%% 2. 修正后的信道（gen_ch 生成，已验证频带正确）
H = load('D:/project/信道产生/data_verify/channel_H_1.mat','channel_H');
h = H.channel_H(:).';
pk = find(abs(h)==max(abs(h)),1);
% 时延扩展：能量累积到 50% / 90% 的位置
e = cumsum(h.^2)/sum(h.^2);
d50 = find(e>=0.5,1) - pk;
d90 = find(e>=0.9,1) - pk;
fprintf('修正信道: %d 点 | peak@%d | max=%.3e | rms=%.3e\n', numel(h), pk, max(abs(h)), rms(h));
fprintf('  时延扩展: 50%%能量在 %d 点内, 90%%在 %d 点内 (%.1f ms)\n', ...
        abs(d50), abs(d90), abs(d90)/24000*1000);

%% 3. 卷积
y = conv(clean24, h);
fprintf('\n卷积结果: %d 点 (%.3f s)\n', numel(y), numel(y)/24000);

%% 4. 频带检查 —— 这是关键
N = 2^nextpow2(numel(y));
P = abs(fft(y,N)).^2; P = P(1:N/2+1);
E = cumsum(P)/sum(P);
lo = find(E>=0.005,1)/N; hi = find(E>=0.995,1)/N;
fprintf('卷积结果 99%% 频带 = %.5f ~ %.5f\n', lo, hi);
if lo > 0.025 && hi < 0.075
    fprintf('  ✓ chirp 成功通过信道（落在信道通带内）\n');
else
    fprintf('  ✗ 频带不对\n');
end

%% 5. 匹配滤波：chirp 出现在哪个延迟
c = ifft( fft(y,N) .* conj(fft([clean24, zeros(1,N-numel(clean24))])) );
c = real(c);
[cm, ci] = max(abs(c(1:numel(y))));
mask = true(1,numel(y)); mask(max(1,ci-300):min(numel(y),ci+300)) = false;
nr = rms(c(mask));
fprintf('匹配滤波: 峰值 %.3e @ 延迟 %d 点（信道主到达在 %d）\n', cm, ci, pk);
fprintf('  峰值/底噪 = %.1f 倍 (%.1f dB)\n', cm/nr, 20*log10(cm/nr));

%% 6. 与旧数据对比
fprintf('\n============ 对比旧数据 ============\n');
old = load('D:/project/te_signal/test_sample_1.mat','noisy_signal');
o = old.noisy_signal(:).';
No = 2^nextpow2(numel(o));
Po = abs(fft(o,No)).^2; Po = Po(1:No/2+1);
Eo = cumsum(Po)/sum(Po);
fprintf('旧含噪信号 99%% 频带 = %.5f ~ %.5f\n', find(Eo>=0.005,1)/No, find(Eo>=0.995,1)/No);
fprintf('  正确应为 0.0333~0.0625，实测偏到了 0.0099~0.0187 那一带\n');

% 旧数据里 chirp 出现在哪
co = ifft( fft(o,No) .* conj(fft([clean, zeros(1,No-numel(clean))])) );
co = real(co);
[com, coi] = max(abs(co(1:numel(o))));
mo = true(1,numel(o)); mo(max(1,coi-300):min(numel(o),coi+300)) = false;
fprintf('旧数据匹配滤波: 峰值 @ 延迟 %d 点，峰值/底噪 = %.1f 倍 (%.1f dB)\n', ...
        coi, com/rms(co(mo)), 20*log10(com/rms(co(mo))));
fprintf('  （延迟 %d ≈ 信道长度 8000，说明信号是"原样+延迟"加进去的，没经过信道）\n', coi);

%% 7. 加噪后 SNR 检查（验证 fix_make_dataset 的加噪逻辑）
fprintf('\n============ 加噪逻辑验证 ============\n');
target_snr = 0;
Py = mean(y.^2);
Pn = Py/10^(target_snr/10);
rng(1);
noisy = y + randn(size(y))*sqrt(Pn);
actual = 10*log10(sum(y.^2)/sum((noisy-y).^2));
fprintf('目标 SNR = %d dB，实测 SNR = %.2f dB\n', target_snr, actual);
