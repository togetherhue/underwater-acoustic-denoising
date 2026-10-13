%% 往返验证（修正版）：确认"取出→重放"是无损的
clc; clear;
fs = 24000; Len = 8000; df = fs/Len;
fl = 798; fh = 1500; fct = 10;
Vfre   = fl:fct:fh;
f_full = fl:df:fh;
bin    = round(f_full/df);

shift_to_2501 = @(x) local_shift(x, Len);

%% 1. 取一个"正确"的信道
V = load('D:/project/信道产生/data_verify/channel_H_1.mat','channel_H');
href = V.channel_H(:).';          % 峰值已在 2501

%% 2. 模拟 bug：把 266:500 的值搬到 80:150
F  = fft(href(:), Len);
Fs = F(round(Vfre/df));           % RAM 实际算的 71 个值
Fb = zeros(1,Len); Fb(80:150) = Fs;
hb = real(ifft(Fb));              % 这就是"放错位置"的信道
hb = shift_to_2501(hb(:).');      % 旧脚本也做了移位

%% 3. 抢救：取出来 → 插值 → 放回 266:500 → 移位
Fb2 = fft(hb(:), Len);
Vr  = Fb2(80:150) * 2;            % ★ 关键：×2 补偿 real() 的减半
Vf2 = interp1(Vfre, real(Vr), f_full,'linear','extrap') + ...
      1i*interp1(Vfre, imag(Vr), f_full,'linear','extrap');
Fc  = zeros(1,Len); Fc(bin) = Vf2;
hc  = real(ifft(Fc));
hc  = shift_to_2501(hc(:).');     % ★ 这一步上次漏了

%% 4. 对比
cc = corrcoef(href(:), hc(:));
fprintf('=== 往返验证 ===\n');
fprintf('原始正确信道 vs 抢救回来的信道\n');
fprintf('  相关系数 = %.6f\n', cc(1,2));
fprintf('  相对误差 = %.6f\n', norm(href(:)-hc(:))/norm(href(:)));

% 只看主瓣附近（远离循环移位边界）
seg = 2400:2600;
fprintf('  主瓣区(2400:2600) 相对误差 = %.6f\n', ...
        norm(href(seg)-hc(seg))/norm(href(seg)));

fprintf('\n  原始 : rms=%.4e  max|h|=%.4e  peak@%d\n', ...
        rms(href), max(abs(href)), find(abs(href)==max(abs(href)),1));
fprintf('  抢救 : rms=%.4e  max|h|=%.4e  peak@%d\n', ...
        rms(hc),   max(abs(hc)),   find(abs(hc)==max(abs(hc)),1));


function y = local_shift(x, Len)
[~,pk] = max(abs(x));
if pk > 2500
    y = x([pk-2500:Len, 1:pk-2501]);
else
    y = x([Len-2500+1+pk:Len, 1:Len-2500+pk]);
end
end
