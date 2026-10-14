function test_stblrnd()
%TEST_STBLRND  验证 α 稳定分布生成器的正确性
%==========================================================================
%  三项验证：
%   1. alpha = 2 必须退化为高斯（峰度 ≈ 3）
%   2. 用特征函数反推 alpha：对 beta=0 有 E[cos(tX)] = exp(-gamma^alpha |t|^alpha)
%      故 ln(-ln E[cos(tX)]) = alpha*ln|t| + alpha*ln(gamma)  -> 斜率即 alpha
%   3. 尾部指数：P(|X|>x) ~ C x^{-alpha}，log-log 斜率应为 -alpha
%  另外检查 beta 的偏斜效果。
%==========================================================================

N = 4e6;
fprintf('样本数 N = %g\n\n', N);

%% ---- 1. alpha = 2 退化为高斯 -----------------------------------------
fprintf('=== 1. alpha = 2 应退化为高斯 N(0, 2*gamma^2) ===\n');
for g = [0.5, 1, 2]
    x = stblrnd(2, 0, g, 0, N, 1);
    fprintf('  gamma=%.2f  样本std=%.4f  理论std=%.4f  峰度=%.3f (理论 3)\n', ...
        g, std(x), sqrt(2)*g, loc_kurt(x));
end

%% ---- 2. 特征函数反推 alpha -------------------------------------------
fprintf('\n=== 2. 特征函数反推 alpha（beta=0, gamma=1）===\n');
fprintf('  %8s %10s %10s %10s\n', '设定alpha', '拟合alpha', '拟合gamma', '相对误差');
for a = [1.2, 1.5, 1.8, 2.0]
    x = stblrnd(a, 0, 1, 0, N, 1);
    tg = 0.05:0.05:1.5;
    C = zeros(size(tg));
    for k = 1:numel(tg)
        C(k) = mean(cos(tg(k) * x));
    end
    C = min(C, 1 - 1e-12); C = max(C, 1e-12);
    y = log(-log(C)); X = log(tg);
    p = polyfit(X, y, 1);
    a_hat = p(1); g_hat = exp(p(2) / a_hat);
    fprintf('  %8.2f %10.4f %10.4f %9.2f%%\n', a, a_hat, g_hat, abs(a_hat-a)/a*100);
end

%% ---- 3. 尾部指数 ------------------------------------------------------
fprintf('\n=== 3. 尾部指数（log-log 生存函数斜率，理论 -alpha）===\n');
fprintf('  %8s %12s %12s\n', '设定alpha', '拟合斜率', '相对误差');
for a = [1.2, 1.5, 1.8]
    x = stblrnd(a, 0, 1, 0, N, 1);
    ax = sort(abs(x));
    qs = [0.99, 0.995, 0.999, 0.9995];
    xs = zeros(size(qs)); ps = zeros(size(qs));
    for k = 1:numel(qs)
        xs(k) = ax(round(qs(k) * numel(ax)));
        ps(k) = 1 - qs(k);
    end
    p = polyfit(log(xs), log(ps), 1);
    fprintf('  %8.2f %12.4f %11.2f%%\n', a, p(1), abs(p(1)+a)/a*100);
end

%% ---- 4. beta 的偏斜效果 -----------------------------------------------
fprintf('\n=== 4. beta 的偏斜效果（alpha=1.5, gamma=1）===\n');
fprintf('  %8s %10s %10s %10s %12s\n', 'beta', 'q05', 'q50', 'q95', '偏斜度(q95+q05-2q50)');
for b = [-1, -0.5, 0, 0.5, 1]
    x = stblrnd(1.5, b, 1, 0, N, 1);
    q = loc_q(x, [0.05, 0.5, 0.95]);
    fprintf('  %8.1f %10.3f %10.3f %10.3f %12.3f\n', b, q(1), q(2), q(3), q(3)+q(1)-2*q(2));
end

%% ---- 5. 与"重尾"的直观对比：超过 10 倍核心宽度的样本比例 --------------
fprintf('\n=== 5. 脉冲性直观对比：|X| 超过其 75%% 分位数 20 倍的样本比例 ===\n');
for a = [1.2, 1.5, 1.8, 2.0]
    x = stblrnd(a, 0, 1, 0, N, 1);
    q75 = loc_q(abs(x), 0.75);
    frac = mean(abs(x) > 20 * q75);
    fprintf('  alpha=%.1f  q75=%.4f  比例=%.2e\n', a, q75, frac);
end
fprintf('\n（alpha 越小，极端离群值越多 —— 这就是"脉冲噪声"的来源）\n');
end

% ===================== 本地工具函数（无工具箱依赖）=====================
function k = loc_kurt(x)
% 样本峰度（超额+3），无工具箱实现
x = x(:) - mean(x);
m2 = mean(x.^2); m4 = mean(x.^4);
k = m4 / m2^2;
end

function q = loc_q(x, p)
% 分位数（线性插值），无工具箱实现
x = sort(x(:));
n = numel(x);
pos = p(:).' * (n - 1) + 1;          % 1-based 位置
lo = floor(pos); hi = ceil(pos);
lo = max(min(lo, n), 1); hi = max(min(hi, n), 1);
w = pos - lo;
q = x(lo) .* (1 - w) + x(hi) .* w;
end
