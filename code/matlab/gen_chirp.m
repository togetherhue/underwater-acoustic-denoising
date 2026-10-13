function s = gen_chirp(fs, T, f0, f1, modtype, env, phi, A)
%GEN_CHIRP  生成扫频信号（源信号）
%==========================================================================
%  fs      : 采样率 [Hz]
%  T       : 时长 [s]
%  f0, f1  : 起止频率 [Hz]
%  modtype : 'LFM' 线性调频 | 'HFM' 双曲调频
%  env     : 'rect' | 'hann' | 'gauss'
%  phi     : 初相 [rad]
%  A       : 幅度
%
%  老师给的 chirp_signal.mat 实测参数（fs=9600）：
%      LFM, 798 -> 1500 Hz, T = 1.0 s, 恒幅, 幅值 ±1
%  即  gen_chirp(9600, 1.0, 798, 1500, 'LFM', 'rect', 0, 1)
%==========================================================================

if nargin < 8 || isempty(A);   A = 1;   end
if nargin < 7 || isempty(phi); phi = 0; end
if nargin < 6 || isempty(env); env = 'rect'; end
if nargin < 5 || isempty(modtype); modtype = 'LFM'; end

t = (0:round(T*fs)-1) / fs;

switch upper(modtype)
    case 'LFM'
        f = f0 + (f1 - f0) * t / T;                       % 线性调频
    case 'HFM'
        f = (f0*f1*T) ./ (f1*T - (f1 - f0)*t);            % 双曲调频
    otherwise
        error('未知调制类型 %s', modtype);
end
ph = 2*pi*cumtrapz(t, f);                                 % 瞬时频率积分得相位

switch lower(env)
    case 'rect',  e = ones(size(t));
    case 'hann',  e = hann(numel(t)).';
    case 'gauss', e = exp(-0.5*((t - mean(t))/(0.25*T)).^2);
    otherwise,    e = ones(size(t));
end

s = A * e .* cos(ph + phi);
end
