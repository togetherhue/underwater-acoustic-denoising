function rescue_ch(srcDirs, outdir, fct)
%RESCUE_CH  把旧 channel_H（bin 放错位置）抢救成正确的信道
%
%   原理：旧脚本 F(fl/fct:fh/fct)=Vpress 把 RAM 算出的 71 个传递函数值
%         放到了 bin 80:150（对应 240~450Hz）。但这 71 个值本身是在
%         798:10:1500 Hz 上算的，物理上正确。取出来放回 bin 266:500 即可。
%
%   用法： rescue_ch({'D:/project/信道产生/data','D:/project/信道产生2/data'}, ...
%                    'D:/project/信道产生/data_fixed', 10)

if nargin < 3 || isempty(fct); fct = 10; end
if nargin < 2 || isempty(outdir)
    outdir = 'D:/project/信道产生/data_fixed';
end
if ~exist(outdir,'dir'); mkdir(outdir); end

fs = 24000; Len = 8000; df = fs/Len;
fl = 798; fh = 1500;
Vfre   = fl:fct:fh;
f_full = fl:df:fh;
bin    = round(f_full/df);
n_exp  = numel(Vfre);

assert(bin(1)==266 && bin(end)==500, 'bin 映射异常');

% 往返验证：用刚生成的正确信道模拟"放错位置"，再抢救回来
vf = 'D:/project/信道产生/data_verify/channel_H_1.mat';
if exist(vf,'file')
    V = load(vf,'channel_H'); href = V.channel_H(:).';
    F  = fft(href(:), Len);
    Fs = F(round(Vfre/df));                 % 模拟 RAM 采样的 71 个值
    Fb = zeros(1,Len); Fb(80:150) = Fs;     % 模拟放错位置
    hb = real(ifft(Fb));
    % --- 抢救 ---
    Fb2 = fft(hb(:), Len);
    Vr  = Fb2(80:150) * 2;
    Vf2 = interp1(Vfre, real(Vr), f_full,'linear','extrap') + ...
          1i*interp1(Vfre, imag(Vr), f_full,'linear','extrap');
    Fc  = zeros(1,Len); Fc(bin) = Vf2;
    hc  = real(ifft(Fc));
    cc = corrcoef(href(:), hc(:));
    fprintf('=== 往返验证 ===\n');
    fprintf('原始正确信道 vs 抢救回来的信道：相关系数 = %.4f\n', cc(1,2));
    fprintf('相对误差 = %.4f\n', norm(href(:)-hc(:))/norm(href(:)));
    fprintf('================\n\n');
end

% 批量抢救
files = {};
for k = 1:numel(srcDirs)
    f = dir(fullfile(srcDirs{k}, 'channel_H_*.mat'));
    for j = 1:numel(f); files{end+1} = fullfile(f(j).folder, f(j).name); end
end
fprintf('待抢救信道：%d 个\n', numel(files));

nok = 0; nbad = 0;
for k = 1:numel(files)
    [~, nm, ~] = fileparts(files{k});
    no = sscanf(nm, 'channel_H_%d');

    S = load(files{k}, 'channel_H');
    h = S.channel_H(:).';
    if numel(h) ~= Len; nbad = nbad + 1; continue; end

    F   = fft(h(:), Len);
    mag = abs(F(1:Len/2+1));
    idx = find(mag > max(mag)*1e-3);

    if numel(idx) ~= n_exp
        nbad = nbad + 1;
        fprintf('  跳过 %s：检出 %d 个频点（期望 %d）\n', nm, numel(idx), n_exp);
        continue;
    end

    Vr    = F(idx) * 2;
    Vfull = interp1(Vfre, real(Vr), f_full,'linear','extrap') + ...
            1i*interp1(Vfre, imag(Vr), f_full,'linear','extrap');
    Fn    = zeros(1,Len); Fn(bin) = Vfull;
    hn    = real(ifft(Fn));

    [~,pk] = max(abs(hn));
    if pk > 2500
        hn = hn([pk-2500:end, 1:pk-2501]);
    else
        hn = hn([Len-2500+1+pk:end, 1:Len-2500+pk]);
    end

    channel_H = hn(:).';
    save(fullfile(outdir, sprintf('channel_H_%d.mat', no)), ...
         'channel_H','fs','Len','fl','fh');
    nok = nok + 1;
end
fprintf('抢救成功 %d 个，跳过 %d 个\n', nok, nbad);

% 自检
fs2 = dir(fullfile(outdir,'channel_H_*.mat'));
npass = 0;
for j = 1:min(numel(fs2), 20)
    S = load(fullfile(outdir, fs2(j).name), 'channel_H');
    P = abs(fft(S.channel_H(:), Len)).^2; P = P(1:Len/2+1);
    E = cumsum(P)/sum(P);
    lo = find(E>=0.005,1)/Len; hi = find(E>=0.995,1)/Len;
    npass = npass + (lo>0.025 && hi<0.075);
    if j <= 5
        fprintf('  %-22s 99%%band = %.5f ~ %.5f\n', fs2(j).name, lo, hi);
    end
end
fprintf('自检：抽查 %d 个，通过 %d 个\n', min(numel(fs2),20), npass);
end
