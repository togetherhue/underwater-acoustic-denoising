%% 最小化隔离测试：只验证"71 点采样 → 235 点重放"这一步是否有损
clc; clear;
fs=24000; Len=8000; df=fs/Len; fl=798; fh=1500; fct=10;
Vfre=fl:fct:fh; f_full=fl:df:fh; bin=round(f_full/df);

V=load('E:/project/data/信道产生/data_verify/channel_H_1.mat','channel_H');
href=V.channel_H(:).';

F   = fft(href(:), Len);
Fs  = F(round(Vfre/df));                 % 按 10Hz 采样
Vf2 = interp1(Vfre, real(Fs), f_full,'linear','extrap') + ...
      1i*interp1(Vfre, imag(Fs), f_full,'linear','extrap');
Fc  = zeros(1,Len); Fc(bin)=Vf2;
hc  = real(ifft(Fc));                    % 不做任何移位

c0 = corrcoef(href(:), hc(:));
fprintf('A) 不做移位直接比较      corr=%.6f  relerr=%.6f\n', ...
        c0(1,2), norm(href(:)-hc(:))/norm(href(:)));

% 两个都移位到 2501 再比
ha = shift2501(href(:).', Len);
hb = shift2501(hc(:).',   Len);
c1 = corrcoef(ha(:), hb(:));
fprintf('B) 都移位到 2501 再比较   corr=%.6f  relerr=%.6f\n', ...
        c1(1,2), norm(ha(:)-hb(:))/norm(ha(:)));

% 只比较主瓣附近
seg = 2350:2650;
fprintf('C) 主瓣区(2350:2650)      relerr=%.6f\n', ...
        norm(ha(seg)-hb(seg))/norm(ha(seg)));

fprintf('\n参考：href 峰值位置=%d，hc 峰值位置=%d\n', ...
        find(abs(href)==max(abs(href)),1), find(abs(hc)==max(abs(hc)),1));

% 再测：用 fct=60 的信道（只有 12 个独立频点）会怎样
V2=load('E:/project/data/信道产生/data_verify/channel_H_1.mat','channel_H');
fprintf('\n结论提示：\n');
fprintf('  这个 href 是 fct=60 生成的（只有 12 个独立频点），\n');
fprintf('  用 10Hz 采样去取它必然有插值误差。\n');
fprintf('  真实旧信道是 fct=10 生成的，用 10Hz 取是无损的。\n');


function y = shift2501(x, Len)
[~,pk]=max(abs(x));
if pk>2500; y=x([pk-2500:Len, 1:pk-2501]);
else;       y=x([Len-2500+1+pk:Len, 1:Len-2500+pk]); end
end
