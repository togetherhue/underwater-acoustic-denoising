function gen_ch(idxs, fct, outdir, seed)
%GEN_CH  生成水声信道冲激响应 channel_H_*.mat（修正版，单进程串行）
%
%   修正内容：原脚本 F(fl/fct:fh/fct) 把 fct 当成了 DFT bin 索引，
%   fct 从 3 改成 10 后通带从 798~1500Hz 掉到 237~447Hz。
%   本函数用 bin = round((fl:df:fh)/df)，与 fct 完全解耦。
%
%   idxs  : 要生成的信道编号列表，例如 1:10
%   fct   : 频率步进 [Hz]
%   outdir: 输出目录
%   seed  : 随机种子（★必须给！否则同时启动的多个进程会生成完全相同的信道）
%
%   用法：  gen_ch(1:10, 10, 'D:/project/信道产生/data_fixed', 1001)

if nargin < 4 || isempty(seed)
    seed = round(mod(now*86400, 1e6));   % 兜底：用时间播种
end
rng(seed, 'twister');
fprintf('随机种子 seed = %d\n', seed);

if nargin < 3 || isempty(outdir)
    outdir = 'D:/project/信道产生/data_fixed';
end

work = 'D:/project/信道产生';
fs = 24000;  Len = 8000;  df = fs/Len;
fl = 798;    fh = 1500;
Dist = 20000;

Vfre   = fl:fct:fh;
f_full = fl:df:fh;
bin    = round(f_full/df);

assert(bin(1) == 266 && bin(end) == 500, 'bin 映射异常');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% exe 放到 ASCII 临时目录，避免中文路径问题
tmp_root = fullfile(tempdir, 'ram_ch');
if ~exist(tmp_root, 'dir'); mkdir(tmp_root); end
exe_asc = fullfile(tmp_root, 'ramgeo1.5_atten.exe');
if ~exist(exe_asc, 'file')
    copyfile(fullfile(work, 'ramgeo1.5_atten.exe'), exe_asc);
end
addpath(work);

fprintf('本进程负责 %d 个信道 | fct=%d Hz (%d 点/信道)\n', ...
        numel(idxs), fct, numel(Vfre));
fprintf('预计 %.1f 分钟\n', numel(idxs)*numel(Vfre)*11.8/60);

t_all = tic;
for ii = idxs
    rng(seed*1000 + ii, 'twister');   % ★每个信道独立播种，保证全局不重复
    td  = tempname(tmp_root);
    mkdir(td);
    old = cd(td);          % old 必须是"进入 td 之前"的目录
    try
        Depth = 70 + 10*randi([1,12]);
        [bath, ssp, p1, p3, p4] = Input_PE(Depth, Dist);
        sd = randi([15, Depth-20]);
        rd = sd + randi([-10,10]);
        Range = bath(1, end);

        Vpress = zeros(1, numel(Vfre));
        for mm = 1:numel(Vfre)
            freq = Vfre(mm);
            dr  = 1500/1600*4;
            dz  = 1500/freq/20;
            wa1 = SeaWater_attenu(freq/1000, 30, 0, 33, 7, 'Thorpe');
            wa  = wa1*1500/freq;
            WriteenvRam(freq, sd, rd, dr, dz, bath, ssp, p1, p3, p4, Range, 8, wa);
            system(exe_asc);
            t0 = load('tl.line');
            Vpress(mm) = t0(end,3) + 1i*t0(end,4);
        end

        Vfull = interp1(Vfre, real(Vpress), f_full, 'linear', 'extrap') + ...
                1i * interp1(Vfre, imag(Vpress), f_full, 'linear', 'extrap');

        F = zeros(1, Len);
        F(bin) = Vfull;
        H = real(ifft(F));

        [~, pk] = max(abs(H));
        if pk > 2500
            H = H([pk-2500:end, 1:pk-2501]);
        else
            H = H([Len-2500+1+pk:end, 1:Len-2500+pk]);
        end

        channel_H = H(:).';
        save(fullfile(outdir, sprintf('channel_H_%d.mat', ii)), ...
             'channel_H', 'fs', 'Len', 'fl', 'fh');
    catch ME
        cd(old);
        try; rmdir(td, 's'); catch; end
        fprintf('  [%s] ch%-3d  失败: %s\n', datestr(now,'HH:MM:SS'), ii, ME.message);
        continue;                       % 单个信道失败不影响后续
    end
    cd(old);                            % ★ 显式切回，不依赖 onCleanup
    try; rmdir(td, 's'); catch; end

    P = abs(fft(channel_H, Len)).^2;  P = P(1:Len/2+1);
    E = cumsum(P)/sum(P);
    lo = find(E>=0.005,1)/Len;
    hi = find(E>=0.995,1)/Len;
    fprintf('  [%s] ch%-3d  ok  99%%band=%.5f~%.5f  (%.1f min)\n', ...
            datestr(now,'HH:MM:SS'), ii, lo, hi, toc(t_all)/60);
end
fprintf('本进程完成，总耗时 %.1f 分钟\n', toc(t_all)/60);
end
