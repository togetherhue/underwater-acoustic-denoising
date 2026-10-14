function test_gen_burst()
%TEST_GEN_BURST  验证突发干扰生成器
%==========================================================================
%  检查项：
%   1. 实际到达次数是否符合泊松期望
%   2. 占空比（受干扰采样点比例）与理论估计是否吻合
%   3. 幅度分布是否落在设定范围内
%   4. mask 与干扰信号是否一致（mask=1 处应有非零干扰，mask=0 处应严格为 0）
%   5. 四种波形类型是否都被生成出来
%   6. 干扰峰值相对干净信号 rms 的倍数（"脉冲噪声比较大"的量化）
%==========================================================================

fs = 24000;
T  = 2.0;                       % 2 秒
N  = round(T * fs);
sig_rms = 1;                    % 干净信号 rms 归一为 1

cfg = struct('rate', 8, 'dur_range', [0.002, 0.030], ...
             'amp_range', [4, 25], 'seed', 20261014, ...
             'type_prob', [0.25, 0.30, 0.25, 0.20]);

fprintf('参数：fs=%d Hz, T=%.1f s, 到达率=%.0f 次/秒, 时长=%.0f~%.0f ms, 幅度=%.0f~%.0f×rms\n\n', ...
    fs, T, cfg.rate, cfg.dur_range(1)*1e3, cfg.dur_range(2)*1e3, ...
    cfg.amp_range(1), cfg.amp_range(2));

n_rep = 200;
cnt = zeros(1, n_rep); duty = zeros(1, n_rep); pmax = zeros(1, n_rep);
types = zeros(1, 4); mask_ok = true;

for r = 1:n_rep
    c = cfg; c.seed = 1000 + r;
    [i, mask, info] = gen_burst(N, fs, c);

    cnt(r)  = numel(info);
    duty(r) = mean(mask);
    pmax(r) = max(abs(i)) / sig_rms;
    for t = 1:numel(info); types(info(t).type) = types(info(t).type) + 1; end

    % ---- mask 与干扰的一致性 ----
    if any(abs(i(mask == 0)) > 0)
        mask_ok = false;
    end
    if any(mask(i ~= 0) == 0)
        mask_ok = false;
    end
    % 幅度范围检查
    amps = [info.amp];
    if any(amps < cfg.amp_range(1) - 1e-9) || any(amps > cfg.amp_range(2) + 1e-9)
        mask_ok = false;
    end
end

fprintf('=== 1. 到达次数 ===\n');
fprintf('  泊松期望 = %.1f 次   实测均值 = %.2f 次   标准差 = %.2f（理论 %.2f）\n', ...
    cfg.rate * T, mean(cnt), std(cnt), sqrt(cfg.rate * T));

fprintf('\n=== 2. 占空比 ===\n');
% 有效平均时长：impulse 类型实际只有 1~3 个采样点，不按 dur_range 走
p_imp = cfg.type_prob(1) / sum(cfg.type_prob);
dur_eff = p_imp * (2/fs) + (1 - p_imp) * mean(cfg.dur_range);
fprintf('  实测占空比 均值 %.4f（%.2f%%）  范围 %.4f ~ %.4f\n', ...
    mean(duty), mean(duty)*100, min(duty), max(duty));
fprintf('  理论估计 rate×E[有效时长] = %.1f×%.4f = %.4f（%.2f%%）\n', ...
    cfg.rate, dur_eff, cfg.rate*dur_eff, cfg.rate*dur_eff*100);

fprintf('\n=== 3. 干扰峰值（相对干净信号 rms）===\n');
fprintf('  均值 %.1f×   中位 %.1f×   范围 %.1f× ~ %.1f×\n', ...
    mean(pmax), median(pmax), min(pmax), max(pmax));
fprintf('  （峰值可超过单次幅度上限 %.0f×，因为多次干扰会叠加）\n', cfg.amp_range(2));

fprintf('\n=== 4. mask 一致性 ===\n');
fprintf('  mask=0 处干扰严格为 0 / mask=1 处干扰非 0 / 幅度在设定范围内 : %s\n', ...
    tern(mask_ok, '✓ 全部通过', '✗ 有异常'));

fprintf('\n=== 5. 波形类型分布（%d 次干扰）===\n', sum(types));
nm = {'尖峰 impulse', '衰减振荡 decay', '限带噪声 burst', '矩形 rect'};
for k = 1:4
    fprintf('  %-18s %6d 次  (%.1f%%)   设定 %.0f%%\n', ...
        nm{k}, types(k), types(k)/sum(types)*100, cfg.type_prob(k)*100);
end

%% ---- 6. 时域与频谱直观展示 -------------------------------------------
c = cfg; c.seed = 20261014;
[i, mask, info] = gen_burst(N, fs, c);
fprintf('\n=== 6. 单次样例（seed=20261014）===\n');
fprintf('  干扰次数 %d   占空比 %.3f%%   峰值 %.1f×rms\n', ...
    numel(info), mean(mask)*100, max(abs(i)));

fprintf('\n  前 8 次干扰明细：\n');
fprintf('    %-8s %-16s %-10s %-8s %s\n', '序号', '波形', '到达(s)', '时长(ms)', '幅度(×rms)');
nm2 = {'impulse','decay','burst','rect'};
for k = 1:min(8, numel(info))
    fprintf('    %-8d %-16s %-10.4f %-8.1f %.1f\n', ...
        k, nm2{info(k).type}, info(k).onset, info(k).dur*1e3, info(k).amp);
end
end

function s = tern(c, a, b)
if c; s = a; else; s = b; end
end
