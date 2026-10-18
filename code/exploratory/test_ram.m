%% 单点连通性测试：验证 RAM exe 能跑通、测单次耗时
work = 'E:/project/data/信道产生';
cd(work);
addpath(work);

Depth = 100; Dist = 20000;
[bath, ssp, p1, p3, p4] = Input_PE(Depth, Dist);
sd = 30; rd = 32;
Range = bath(1, end);

freq = 1000;
dr  = 1500/1600*4;
dz  = 1500/freq/20;
wa1 = SeaWater_attenu(freq/1000, 30, 0, 33, 7, 'Thorpe');
wa  = wa1*1500/freq;

t1 = tic;
WriteenvRam(freq, sd, rd, dr, dz, bath, ssp, p1, p3, p4, Range, 8, wa);
tw = toc(t1);

t2 = tic;
[st, out] = system('ramgeo1.5_atten.exe');
tr = toc(t2);

fprintf('WriteenvRam  : %.2f s\n', tw);
fprintf('RAM exe      : %.2f s   (exit=%d)\n', tr, st);
fprintf('exe 输出前 200 字符: %s\n', out(1:min(200,numel(out))));

if exist('tl.line', 'file')
    A = load('tl.line');
    fprintf('tl.line      : %d x %d\n', size(A,1), size(A,2));
    fprintf('Vpress(1000Hz) = %.6g + %.6gi   |V| = %.6g\n', ...
            A(end,3), A(end,4), hypot(A(end,3), A(end,4)));
else
    fprintf('!! 没生成 tl.line\n');
end
