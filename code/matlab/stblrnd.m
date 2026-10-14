function x = stblrnd(alpha, beta, gamma, delta, m, n)
%STBLRND  α 稳定分布随机数生成器（Chambers-Mallows-Stuck 方法）
%==========================================================================
%  生成 X ~ S(alpha, beta, gamma, delta)
%
%  alpha : 特征指数，0 < alpha <= 2
%            alpha = 2  -> 高斯
%            alpha = 1  -> 柯西
%            alpha 越小，尾部越重（脉冲性越强）
%  beta  : 偏斜参数，-1 <= beta <= 1（beta=0 为对称分布 SαS）
%  gamma : 尺度参数，gamma > 0（注意：不是标准差！alpha<2 时方差不存在）
%  delta : 位置参数
%  m, n  : 输出尺寸（默认 1x1）
%
%  参考：Chambers, Mallows & Stuck (1976), "A method for simulating stable
%        random variables", JASA 71(354): 340-344.
%
%  【重要】alpha < 2 时二阶矩不存在（方差无穷），因此不能用"噪声功率"
%          来定义信噪比。参见 gen_noise_pam.m 中的 75% 分位数匹配方案。
%==========================================================================

if nargin < 6 || isempty(n); n = 1; end
if nargin < 5 || isempty(m); m = 1; end
if nargin < 4 || isempty(delta); delta = 0; end
if nargin < 3 || isempty(gamma); gamma = 1; end
if nargin < 2 || isempty(beta);  beta  = 0; end

assert(alpha > 0 && alpha <= 2, 'alpha 必须落在 (0, 2]');
assert(abs(beta) <= 1, 'beta 必须落在 [-1, 1]');
assert(gamma > 0, 'gamma 必须为正');

% ---- alpha = 2：直接退化为高斯，避免 tan(pi*alpha/2) 的数值问题 --------
if abs(alpha - 2) < 1e-12
    x = sqrt(2) * gamma * randn(m, n) + delta;      % S(2,b,g,d) = N(d, 2g^2)
    return
end

% ---- CMS 方法 ----------------------------------------------------------
% V ~ U(-pi/2, pi/2)，W ~ Exp(1)
V = (rand(m, n) - 0.5) * pi;
V = max(min(V, pi/2 - 1e-12), -pi/2 + 1e-12);       % 防止 cos(V) = 0
U = rand(m, n);
U = max(U, realmin);                                 % 防止 log(0)
W = -log(U);

if abs(alpha - 1) > 1e-9
    % ---- alpha != 1 ----
    zeta = -beta * tan(pi * alpha / 2);
    xi   = atan(zeta) / alpha;
    X = (1 + zeta^2)^(1 / (2 * alpha)) ...
        .* (sin(alpha * (V + xi)) ./ cos(V).^(1 / alpha)) ...
        .* (cos(V - alpha * (V + xi)) ./ W).^((1 - alpha) / alpha);
    x = gamma * X + delta;
else
    % ---- alpha == 1（柯西族）----
    xi = pi / 2;
    X = (2 / pi) * ((pi / 2 + beta * V) .* tan(V) ...
        - beta * log((pi / 2 * W .* cos(V)) ./ (pi / 2 + beta * V)));
    x = gamma * X + delta + (2 / pi) * beta * gamma * log(gamma);
end
end
