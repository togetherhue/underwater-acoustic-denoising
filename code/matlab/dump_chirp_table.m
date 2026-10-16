function dump_chirp_table(out_csv)
%DUMP_CHIRP_TABLE  导出 chirp 参数表到 CSV
%======================================================================
%  生成逻辑与 make_dataset_pam.m / make_dataset_gen.m 第 2 节【完全一致】
%  （同 seed、同 rand 调用顺序）。导出后供 Python 侧做"波形参数 vs 效应"
%  的相关分析，避免手工转录出错。
%
%  用法：dump_chirp_table('D:/project/chirp_table.csv')
%======================================================================
if nargin < 1 || isempty(out_csv)
    out_csv = 'D:/project/chirp_table.csv';
end

seed = 20261014;          % 与 make_dataset_pam.m 的 cfg.seed 一致
n_total = 20;

rng(seed, 'twister');
P = struct('f0',{},'f1',{},'T',{},'env',{},'phi',{},'A',{});
P(1).f0 = 798;  P(1).f1 = 1500;  P(1).T = 1.0;
P(1).env = 'rect';  P(1).phi = 0;  P(1).A = 1;
for i = 2:n_total
    a = 800  + 250*rand();
    b = 1280 + 219*rand();
    if rand() < 0.5; P(i).f0 = a;  P(i).f1 = b;
    else;            P(i).f0 = b;  P(i).f1 = a; end
    P(i).T = 0.6 + 0.4*rand();
    r = rand();
    if r < 0.5;      P(i).env = 'rect';
    elseif r < 0.75; P(i).env = 'hann';
    else;            P(i).env = 'gauss'; end
    P(i).phi = 2*pi*rand();
    P(i).A   = 0.5 + 0.5*rand();
end

fid = fopen(out_csv, 'w');
fprintf(fid, 'chirp_id,f0,f1,T,env,phi,A,bw,f_lo,f_hi\n');
for i = 1:n_total
    lo = min(P(i).f0, P(i).f1);
    hi = max(P(i).f0, P(i).f1);
    fprintf(fid, '%d,%.6f,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        i, P(i).f0, P(i).f1, P(i).T, P(i).env, P(i).phi, P(i).A, ...
        abs(P(i).f1 - P(i).f0), lo, hi);
end
fclose(fid);
fprintf('chirp 参数表 -> %s\n', out_csv);
fprintf('  带宽范围 %.1f ~ %.1f Hz，中位 %.1f Hz\n', ...
    min(abs([P.f1] - [P.f0])), max(abs([P.f1] - [P.f0])), ...
    median(abs([P.f1] - [P.f0])));
end
