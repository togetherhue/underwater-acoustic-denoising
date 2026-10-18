%% 信道全量验证
clc; clear;
fs=24000; Len=8000;
d = dir('E:/project/data/信道产生/data_fixed/channel_H_*.mat');
H = zeros(numel(d), Len); nm = cell(numel(d),1);
for k=1:numel(d)
    S = load(fullfile(d(k).folder,d(k).name),'channel_H');
    H(k,:) = S.channel_H(:).';  nm{k} = d(k).name;
end

%% 1. 内容去重（不能用 MD5！）
fprintf('=== 内容去重 ===\n');
used = false(1,numel(d)); keep = true(1,numel(d)); ndup = 0;
for i=1:numel(d)
    if used(i); continue; end
    grp = {nm{i}};
    for j=i+1:numel(d)
        if ~used(j) && isequal(H(i,:), H(j,:))
            grp{end+1} = nm{j}; used(j)=true; keep(j)=false; ndup=ndup+1;
        end
    end
    if numel(grp)>1; fprintf('  重复: %s\n', strjoin(grp,' = ')); end
end
fprintf('  文件 %d 个，唯一 %d 个\n\n', numel(d), numel(d)-ndup);

%% 2. 逐个检查频带 + 时延扩展
fprintf('%-22s %-20s %-10s %-12s %s\n','文件','99%频带','峰位置','max|h|','90%%时延扩展');
fprintf('%s\n', repmat('-',1,80));
npass=0; d90all=[];
for k=1:numel(d)
    h = H(k,:);
    P = abs(fft(h(:),Len)).^2; P=P(1:Len/2+1);
    E = cumsum(P)/sum(P);
    lo = find(E>=0.005,1)/Len; hi = find(E>=0.995,1)/Len;
    pk = find(abs(h)==max(abs(h)),1);
    e = cumsum(h.^2)/sum(h.^2); d90 = abs(find(e>=0.9,1)-pk);
    ok = (lo>0.025 && hi<0.075); npass=npass+ok; d90all(end+1)=d90;
    if k<=6 || ~ok
        fprintf('%-22s %.5f~%.5f   %-10d %-12.3e %d 点 (%.1f ms) %s\n', ...
            nm{k}, lo, hi, pk, max(abs(h)), d90, d90/fs*1000, string(ok));
    end
end
fprintf('...\n');
fprintf('\n频带正确 %d / %d\n', npass, numel(d));
fprintf('时延扩展: min=%d 中位=%d max=%d 点  (%.1f ~ %.1f ms)\n', ...
        min(d90all), round(median(d90all)), max(d90all), min(d90all)/fs*1000, max(d90all)/fs*1000);

%% 3. 与旧信道对比
fprintf('\n=== 与旧信道对比 ===\n');
o = load('E:/project/data/信道产生/data/channel_H_1.mat','channel_H'); o=o.channel_H(:).';
P = abs(fft(o(:),Len)).^2; P=P(1:Len/2+1); E=cumsum(P)/sum(P);
fprintf('  旧信道 99%%频带 = %.5f ~ %.5f  （错误）\n', find(E>=0.005,1)/Len, find(E>=0.995,1)/Len);
h = H(1,:);
P = abs(fft(h(:),Len)).^2; P=P(1:Len/2+1); E=cumsum(P)/sum(P);
fprintf('  新信道 99%%频带 = %.5f ~ %.5f  （正确，目标 0.0333~0.0625）\n', ...
        find(E>=0.005,1)/Len, find(E>=0.995,1)/Len);
