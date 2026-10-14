function [i, mask, info] = gen_burst(N, fs, cfg)
%GEN_BURST  突发脉冲干扰生成器
%==========================================================================
%  按老师方案「随机幅度、发生概率、持续时间和突发间隔」建模：
%
%      i(t) = Σ_k A_k · w_k(t - τ_k) · 1{0 <= t-τ_k < d_k}
%
%  - 到达时刻 τ_k：泊松过程（平均到达率 cfg.rate 次/秒）
%  - 幅度 A_k   ：随机，相对干净信号 rms 归一
%  - 持续时间 d_k：随机
%  - 波形 w_k   ：四种类型，覆盖老师提到的"尖峰、脉冲和突发边缘"
%
%  输入：
%    N   采样点数
%    fs  采样率
%    cfg 结构体，字段见下方默认值
%
%  输出：
%    i     干扰信号（与信号同长度）
%    mask  干扰位置掩码（1 = 该采样点受干扰）—— 供 L_imp 辅助损失使用
%    info  每次干扰的明细（onset/dur/amp/type），便于核查与生成弱标签
%
%  注：老师文档明确「真实海试数据缺少精确标签时，不应直接把人工阈值检测
%      结果当作绝对正确的监督标签」——所以 mask 只在仿真数据中作为真值，
%      实测数据上只能作弱监督。
%==========================================================================

if nargin < 3 || isempty(cfg); cfg = struct(); end

def = struct( ...
    'rate',       5, ...                      % 平均到达率 [次/秒]
    'dur_range',  [0.002, 0.030], ...         % 持续时间范围 [s]
    'amp_range',  [4, 25], ...                % 峰值幅度范围（× rms(signal)）
    'type_prob',  [0.25, 0.30, 0.25, 0.20], ... % [impulse decay burst rect]
    'freq_range', [200, 2600], ...            % burst/rect 的载频范围 [Hz]
    'bw_frac',    [0.15, 0.60], ...           % 相对载频的带宽比例
    'seed',       []);
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(cfg, fn{k}); cfg.(fn{k}) = def.(fn{k}); end
end
if ~isempty(cfg.seed); rng(cfg.seed, 'twister'); end

T = N / fs;
i = zeros(1, N);
mask = zeros(1, N);

% ---- 泊松到达 ----------------------------------------------------------
n_burst = poissrnd_loc(cfg.rate * T);
info = struct('onset', {}, 'dur', {}, 'amp', {}, 'type', {}, 'nsamp', {});

for k = 1:n_burst
    tau = rand() * T;                                  % 到达时刻 [s]
    d   = cfg.dur_range(1) + diff(cfg.dur_range) * rand();
    A   = cfg.amp_range(1) + diff(cfg.amp_range) * rand();

    % 按概率选波形类型
    r = rand();
    c = cumsum(cfg.type_prob / sum(cfg.type_prob));
    ty = find(r <= c, 1, 'first');
    if isempty(ty); ty = numel(c); end

    n0 = max(1, round(tau * fs) + 1);
    switch ty
        case 1   % ---- 尖峰：极短，1~3 个采样点 ----
            ns = 1 + floor(rand() * 3);
            w = ones(1, ns);
            w = w * (2 * (rand() > 0.5) - 1);          % 随机极性
        case 2   % ---- 衰减振荡：阻尼正弦（瞬态）----
            ns = max(3, round(d * fs));
            fc = cfg.freq_range(1) + diff(cfg.freq_range) * rand();
            tt = (0:ns-1) / fs;
            tau_d = d / (2 + 3 * rand());              % 衰减时间常数
            w = exp(-tt / tau_d) .* sin(2 * pi * fc * tt + 2 * pi * rand());
            w = w / max(abs(w));                        % 归一后由 A 控制幅度
        case 3   % ---- 限带噪声突发 ----
            ns = max(3, round(d * fs));
            fc = cfg.freq_range(1) + diff(cfg.freq_range) * rand();
            bw = fc * (cfg.bw_frac(1) + diff(cfg.bw_frac) * rand());
            lo = max(fc - bw / 2, 20) / (fs / 2);
            hi = min(fc + bw / 2, fs / 2 * 0.98) / (fs / 2);
            if hi <= lo; hi = min(lo * 1.5, 0.98); end
            [b, a] = butter(4, [lo, hi], 'bandpass');
            w = filter(b, a, randn(1, ns));
            w = w / max(abs(w));
        otherwise % ---- 矩形包络窄带段 ----
            ns = max(3, round(d * fs));
            fc = cfg.freq_range(1) + diff(cfg.freq_range) * rand();
            tt = (0:ns-1) / fs;
            w = sin(2 * pi * fc * tt + 2 * pi * rand());
    end

    idx = n0 : min(N, n0 + numel(w) - 1);
    if isempty(idx); continue; end
    seg = w(1:numel(idx));
    i(idx) = i(idx) + A * seg;
    mask(idx) = 1;

    info(end+1) = struct('onset', tau, 'dur', numel(idx) / fs, 'amp', A, ...
                         'type', ty, 'nsamp', numel(idx)); %#ok<AGROW>
end
end

% ------------------------------------------------------------------------
function n = poissrnd_loc(lambda)
% 泊松随机数（无工具箱依赖），lambda 可较大时用正态近似兜底
if lambda <= 0
    n = 0; return
end
if lambda < 30
    % Knuth 算法
    L = exp(-lambda); k = 0; p = 1;
    while true
        k = k + 1;
        p = p * rand();
        if p <= L; n = k - 1; return; end
    end
else
    n = max(0, round(lambda + sqrt(lambda) * randn()));
end
end
