%% 诊断：打印每一步的尺寸和数值
clc; clear;
fs=24000; Len=8000; df=fs/Len; fl=798; fh=1500;
Vfre = fl:10:fh;          % 798..1498, 71 个
f_full = fl:df:fh;        % 798..1500, 235 个
bin = round(f_full/df);   % 266..500

V=load('E:/project/data/信道产生/data_verify/channel_H_1.mat','channel_H');
href=V.channel_H(:).';
fprintf('href size=%s  peak@%d  max=%.4e\n', mat2str(size(href)), ...
        find(abs(href)==max(abs(href)),1), max(abs(href)));

% --- 1. 从 href 的频谱里按 10Hz 采样 ---
F  = fft(href(:), Len);            % 8000x1
idx = round(Vfre/df);              % 1x71
Fs = F(idx);                       % 71x1
fprintf('size F=%s  idx=%s  Fs=%s\n', mat2str(size(F)), mat2str(size(idx)), mat2str(size(Fs)));

% --- 2. 直接比较：href 在 bin 上的值  vs  Fs ---
F0 = F(bin);                       % 235x1, href 频谱在 266:500 的值
fprintf('\nF0(1:4)  = %s\n', mat2str(real(F0(1:4))', 5));
fprintf('Fs(1:4)  = %s\n', mat2str(real(Fs(1:4))', 5));
fprintf('Fs 应等于 F0 在 idx 位置的值：\n');
fprintf('F0(idx(1:4)) = %s\n', mat2str(real(F0(idx(1:4)-265))', 5));

% --- 3. 插值回 235 点 ---
Vf2 = interp1(Vfre(:), real(Fs), f_full(:), 'linear', 'extrap') + ...
      1i*interp1(Vfre(:), imag(Fs), f_full(:), 'linear', 'extrap');
fprintf('\nsize Vf2=%s\n', mat2str(size(Vf2)));
fprintf('Vf2(1:4) = %s\n', mat2str(real(Vf2(1:4))', 5));
fprintf('相对误差 ||Vf2-F0||/||F0|| = %.6f\n', norm(Vf2(:)-F0(:))/norm(F0(:)));

% --- 4. 重建 ---
Fc = zeros(Len,1); Fc(bin) = Vf2;
hc = real(ifft(Fc));
fprintf('\nhc size=%s  peak@%d  max=%.4e\n', mat2str(size(hc)), ...
        find(abs(hc)==max(abs(hc)),1), max(abs(hc)));
fprintf('href(1:4) = %s\n', mat2str(href(1:4), 5));
fprintf('hc(1:4)   = %s\n', mat2str(hc(1:4)', 5));
cc = corrcoef(href(:), hc(:));
fprintf('corr(href, hc) = %.6f\n', cc(1,2));

% --- 5. 检验：如果不用 href（已移位），改用未移位的频谱 ---
% 去掉移位：先找出 href 相对 h0 的移位量
% href 峰值在 2500，说明 h0 峰值在 2500 - s ... 无法直接恢复
% 改用一个更干净的验证：直接构造一个已知频谱
Ftest = zeros(Len,1);
Vknown = (1:235)' + 1i*(235:-1:1)';
Ftest(bin) = Vknown;
h0 = real(ifft(Ftest));
F0b = fft(h0, Len);
fprintf('\n=== 干净验证（不涉及移位）===\n');
fprintf('h0 peak@%d\n', find(abs(h0)==max(abs(h0)),1));
Fsb = F0b(idx);
Vf2b = interp1(Vfre(:), real(Fsb), f_full(:),'linear','extrap') + ...
       1i*interp1(Vfre(:), imag(Fsb), f_full(:),'linear','extrap');
Fcb = zeros(Len,1); Fcb(bin) = Vf2b;
hcb = real(ifft(Fcb));
fprintf('Vknown(idx-265)(1:4) = %s\n', mat2str(real(Vknown(idx(1:4)-265))', 5));
fprintf('Fsb(1:4)             = %s\n', mat2str(real(Fsb(1:4))', 5));
fprintf('Vf2b(1:4)            = %s\n', mat2str(real(Vf2b(1:4))', 5));
fprintf('Vknown(1:4)          = %s\n', mat2str(real(Vknown(1:4))', 5));
cc2 = corrcoef(h0(:), hcb(:));
fprintf('corr(h0, hcb) = %.6f\n', cc2(1,2));
