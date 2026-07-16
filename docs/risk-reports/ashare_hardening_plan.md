# ashare-lab 系统安全加固计划

**版本**: v1.0
**日期**: 2026-07-15
**编制人**: Minxi Hou
**状态**: 送审

---

## 一、背景与目标

### 1.1 背景

ashare-lab 是一个 A 股模拟盘交易系统，基于 TRA 神经网络 + ALSTM 模型，覆盖 CSI1000（约 2597 只小盘股）。系统自 2026 年 6 月上线运行，当前净值约 28 万 RMB，累计收益 -6.34%，最大回撤 7.77%。

2026 年 7 月 15 日，对系统进行了五维度极限场景攻击测试（地缘政治、流动性、监管政策、模型算法、运维基础设施），发现 12 项风险，其中 2 项为致命级（P0），5 项为高危（P1），5 项为中危（P2）。

### 1.2 目标

在真钱上线前，完成全部 P0 和 P1 风险的修复与验证，确保系统在以下极端场景下不会发生不可控的资金损失：

- 千股跌停（2024 年 1-2 月 CSI1000 股灾场景）
- 政策突变（IPO 暂停、量化限制、行业整顿）
- 模型失效（IC 归零、因子拥挤、风格反转）
- 基础设施故障（GPU 宕机、数据库损坏、pipeline 双跑）

### 1.3 不在本计划范围内

- QMT 实盘接入（等券商审批通过后另立计划）
- 模型重训频率优化（已通过 model_age_days 门控解决）
- iLink 推送路径（已通过 hermes-gateway 修复）
- 回测引擎改造（当前引擎满足模拟盘需求）

---

## 二、现状评估

### 2.1 已完成的修复（2026-07-15 本次 session）

| 修复 | 文件 | 内容 | 状态 |
|------|------|------|------|
| H1 跌停退出 | engine.py, ledger.py | carry_day 重置 + reset_count 跟踪 + max_resets=5 | 已合并 |
| H2 报告门控 | pipeline.py | 大涨静默，大跌 CRASH ALERT + alert.py 推送 | 已合并 |
| H3 强制平仓 | risk.py | 15% 回撤时强卖全部仓位 | 已合并 |
| H4 Pipeline 互斥 | ashare-pipeline.sh | flock FD 8，60s 超时 | 已合并 |
| meta.json 生成 | predict.py, ashare-pipeline.sh | model_age_days + ic 字段，SCP 回传 | 已合并 |
| iLink 推送 | report.py | 走 hermes-gateway HTTP API | 已合并 |
| 模型重训门控 | ashare-retrain.sh | model_age_days 驱动，每日 22:00 检查 | 已合并 |

### 2.2 待修复的风险

| 编号 | 风险 | 严重度 | 当前状态 | 详细方案 |
|------|------|--------|----------|----------|
| R1 | 跌停锁仓（max_resets 后仍无退出） | P0 | 部分修复（5 轮重置） | solution_R1.md |
| R4 | 模型在噪声上交易（IC 永远 null） | P1 | 无监控 | solution_R4.md |
| R4b | 中性化在危机中失效（OLS 不鲁棒） | P1 | 无鲁棒回归 | solution_R4b.md |
| R5 | 数据库损坏静默丢失 | P0 | 无完整性检查 | solution_R5.md |
| R6 | 隐藏滑点（固定 0.1% vs 实际 2-5%） | P1 | 无动态模型 | solution_R6.md |
| R7 | 停牌无限期持仓 | P1 | 无持仓上限 | solution_R7.md |
| R8 | 因子拥挤（Alpha158 公开特征） | P2 | 无拥挤检测 | solution_R8.md |
| R9 | 风格反转（3 年训练窗口锁定） | P2 | 无风格监控 | solution_R9.md |
| R11 | 过期预测无截断 | P0 | 无年龄阈值 | solution_R11.md |
| R12 | 预测坍缩不可检测 | P1 | 无多样性检查 | solution_R12.md |

### 2.3 已修复的风险（本次 session 之前）

| 编号 | 风险 | 修复时间 |
|------|------|----------|
| R2 | 报告门控吞噬暴跌警报 | 2026-07-15（H2） |
| R3 | 回撤冻结无强平 | 2026-07-15（H3） |
| R10 | Pipeline 双跑 | 2026-07-15（H4） |

---

## 三、实施计划

### Phase 1: 安全网（阻断不可控损失）

**目标**: 确保系统在任何单一故障下不会发生超过 10% NAV 的不可控损失。

**工期**: 约 12 小时（2-3 个工作日）

| 序号 | 任务 | 涉及文件 | 工作量 | 依赖 | 验收标准 |
|------|------|----------|--------|------|----------|
| 1.1 | R5: 数据库完整性检查 + 黄金备份 | ledger.py, pipeline.py | 2-3h | 无 | PRAGMA integrity_check 在备份前执行；月度黄金备份 chmod 444 不可覆盖 |
| 1.2 | R11: 过期预测 3 天硬截断 | ashare-pipeline.sh, risk.py | 4h | 无 | stale parquet 超过 3 天则拒绝使用，触发 alert |
| 1.3 | R1: 跌停永不取消卖单 | engine.py | 2-3h | 1.1 | 取消 max_resets 限制，卖单永久 carry 直到成交 |
| 1.4 | R12: 预测坍缩三层检测 | predict.py | 2.5h | 无 | score_std < 0.001 时拒绝写 parquet |

**Gate 1 验收标准**:
- 全部 4 项任务代码完成
- 3 轮 forge review 零 confirmed findings
- 全量测试通过（682 passed）
- 模拟盘连续运行 7 天无异常

### Phase 2: 信号可信度（建立对模型的信任）

**目标**: 系统能够实时感知模型是否在产生有意义的信号，并在信号退化时自动降级。

**工期**: 约 16 小时（3-4 个工作日）

| 序号 | 任务 | 涉及文件 | 工作量 | 依赖 | 验收标准 |
|------|------|----------|--------|------|----------|
| 2.1 | R4: PSI 预测分布监控 | predict.py, pipeline.py | 8h | Phase 1 | PSI > 0.25 时触发告警；每日记录到 meta.json |
| 2.2 | R4: T+5 滞后 IC 计算 | pipeline.py | 4h | 2.1 | 5 天滚动 IC 写入 reports 表；IC < 0.01 触发告警 |
| 2.3 | R4b: Huber 鲁棒中性化 | neutralize.py | 8h | 无 | OLS 替换为 Huber M-估计；R2 和残差正态性写入日志 |
| 2.4 | R6: Almgren-Chriss 动态滑点 | engine.py, pipeline.py | 3h | 无 | 滑点 = f(波动率, 参与率)；平静市场约 0.1%，危机约 2-5% |

**Gate 2 验收标准**:
- 全部 4 项任务代码完成
- 3 轮 forge review 零 confirmed findings
- PSI 监控运行 30 天，数据完整
- 滞后 IC 持续 > 0.01（5 天滚动）
- 至少经历 1 个交易日跌幅 > 3%，验证强平 + 报告 + 滑点模型联动

### Phase 3: 高级保护（长期韧性）

**目标**: 系统能够应对因子拥挤、风格反转、停牌等慢变量风险。

**工期**: 约 50+ 小时（4-6 周）

| 序号 | 任务 | 涉及文件 | 工作量 | 依赖 | 验收标准 |
|------|------|----------|--------|------|----------|
| 3.1 | R7: 停牌风险评分 + 25 天持仓上限 | engine.py, risk.py | 13-20h | Phase 1 | 停牌超过 25 天自动取消；高风险股集中度 <= 5% |
| 3.2 | R8: 因子拥挤指数（ETF 流量代理） | 新模块 | 13-20 天 | 无 | FCI > 1.5 sigma 时机械因子暴露减半 |
| 3.3 | R9: HMM 风格反转检测 + 自适应混合 | blend.py, pipeline.py | 18 天 | 2.1 | 熊市时 ML 权重降至 0.45；风格反转检测延迟 <= 3 天 |

**Gate 3 验收标准**:
- 全部 3 项任务代码完成
- 3 轮 forge review 零 confirmed findings
- 模拟盘连续运行 60 天，零次不可控持仓（停牌/跌停）
- 真钱上线前通过 Gate 3 全部检查项

---

## 四、验收标准总表

### 4.1 真钱上线前置条件（Gate 1 + Gate 2 必须全部通过）

| 检查项 | 标准 | 验证方法 |
|--------|------|----------|
| 跌停退出 | 卖单永不取消，永久 carry 直到成交 | 注入 3 天跌停数据，验证卖单状态 |
| 数据库保护 | 备份前执行完整性检查 | 注入损坏 DB，验证检测 + 恢复 |
| 过期预测 | 超过 3 天的 parquet 被拒绝 | 用 10 天前的文件跑 pipeline，验证拒绝 |
| 预测坍缩 | 全零分数被拒绝写入 | 注入全零预测，验证 ValueError |
| 信号质量 | PSI 持续 < 0.25，滞后 IC > 0.01 | 30 天运行数据 |
| 强制平仓 | 15% 回撤时全部仓位强卖 | 注入 16% 回撤数据，验证卖单生成 |
| 报告投递 | 每个交易日微信报告送达 | 30 天零遗漏 |
| Pipeline 互斥 | 两次并发执行，第二次被拒绝 | 手动触发两次，验证第二次 exit 1 |
| Crash Alert | NAV 跌 >20% 时 alert.py 推送 | 注入 25% 跌幅数据，验证 alert 触发 |

### 4.2 毕业标准（Gate 3 通过后可上线真钱）

| 检查项 | 标准 |
|--------|------|
| 模拟盘运行天数 | >= 60 个交易日 |
| 信号质量 | 滞后 IC 持续 > 0.01（20 天滚动） |
| 危机验证 | 至少经历 1 次单日跌幅 > 3%，风险控制正确触发 |
| 零不可控持仓 | 停牌/跌停持仓均有退出计划 |
| 报告完整性 | 零遗漏，包括跌幅日 |

---

## 五、时间线

2026-07-15  P0 加固已完成（H1-H4）

Phase 1 开始（7/16）
  1.1 DB 完整性（7/16，2-3h）
  1.2 过期截断（7/16，4h）
  1.3 跌停永不取消（7/17，2-3h）
  1.4 坍缩检测（7/17，2.5h）

Gate 1 验收（7/18）
  模拟盘开始 7 天验证期

Phase 2 开始（7/25）
  2.1 PSI 监控（7/25-26，8h）
  2.2 滞后 IC（7/27，4h）
  2.3 Huber 中性化（7/28-29，8h）
  2.4 动态滑点（7/30，3h）

Gate 2 验收（7/31）
  模拟盘开始 30 天验证期

Phase 3 开始（8/1，与月度重训同步）
  3.1 停牌风险（8/1-8/5，13-20h）
  3.2 因子拥挤（8/6-8/20，13-20 天）
  3.3 风格反转（8/21-9/7，18 天）

Gate 3 验收（9/8）
  60 天验证期完成

真钱上线（9/8 之后，等 Gate 3 全部通过）

**总工期**: 约 8 周（2026-07-16 至 2026-09-08）

---

## 六、回滚方案

### 6.1 代码回滚

所有修复通过 git worktree + feature branch 开发，合并前经过 3 轮 forge review。如发现问题：

回滚单个 commit: git revert <commit-hash>
回滚整个 phase: git revert <first-commit>..<last-commit>

### 6.2 数据库回滚

- 每日 hot_backup 保留 30 天
- Phase 1 新增月度黄金备份（chmod 444，不可覆盖）
- 恢复优先级：当日备份 -> 最近有效备份 -> 黄金备份

### 6.3 模型回滚

- latest.pt 符号链接指向最新训练模型
- 每次训练保留历史 w*.pt 文件
- 回滚：ln -sf w9.pt latest.pt（指向已知有效模型）

### 6.4 紧急熔断

- 手动停止 pipeline: systemctl --user stop ashare-pipeline.timer
- 手动停止重训: systemctl --user stop ashare-retrain.timer
- 清空挂单: SQLite 直接操作 orders 表

---

## 七、风险与依赖

### 7.1 计划风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| forge review 发现重大设计缺陷 | 中 | Phase 延期 2-3 天 | 每个 Phase 预留 1 天 buffer |
| A 股市场在验证期发生极端事件 | 低 | 验证期延长 | 极端事件本身就是最好的验证 |
| QMT 审批延迟 | 中 | 真钱上线推迟 | 模拟盘可独立运行，不阻塞加固 |
| GPU 故障导致训练中断 | 低 | Phase 3 延期 | retrain timer 有 WOL + 重试逻辑 |

### 7.2 外部依赖

| 依赖 | 当前状态 | 影响 |
|------|----------|------|
| QMT 审批 | 等待中（2-3 工作日） | 真钱上线必须 |
| win-gpu 可用性 | 正常 | 每日预测必须 |
| X500 可用性 | 正常 | pipeline 必须 |
| hermes-gateway | 在线 | iLink 推送必须 |

---

## 八、附录

### 8.1 报告清单

全部报告位于 docs/risk-reports/ 和 docs/research/：

**攻击报告**（5 份）:
- attack_geopolitical.md — 地缘/战争/黑天鹅
- attack_liquidity.md — 流动性/微观结构
- attack_regulatory.md — 监管/政策冲击
- attack_model.md — 模型/算法/数据
- attack_operational.md — 运维/基础设施

**综合报告**（2 份）:
- ashare_risk_report.md — 综合风险评估（含风险矩阵、一天归零分析）
- blind_spots_research.md — 12 盲区深度调研（含行业标准、实现方案）

**解决方案**（10 份）:
- solution_R1.md 至 solution_R12.md — 每个风险的具体解决方案

**调研报告**（6 份）:
- draft_codebase_tools_comparison.md — 代码智能工具对比
- draft_exhaustive_code_tools_search.md — 59 工具清单
- draft_shell_crossfile_research.md — Shell 跨文件调用
- draft_fleet_code_intel_briefing.md — 舰队长决策简报
- draft_qmt_application_guide.md — QMT 申请指南
- draft_retraining_research_report.md — 重训频率调研

### 8.2 代码变更清单

本次 session 修改的文件：
- ashare_lab/paper/engine.py — 跳停退出 + reset_count
- ashare_lab/paper/ledger.py — orders 表 reset_count 列
- ashare_lab/paper/pipeline.py — 报告门控 + crash alert + IC 追踪
- ashare_lab/paper/risk.py — 回撤强制平仓
- ashare_lab/research/predict.py — meta.json 生成
- ashare_lab/research/train.py — 双 GPU nvidia-smi 修复
- ashare_lab/research/supplementary.py — latest.pt 符号链接
- scripts/ashare-pipeline.sh — flock 互斥 + meta.json SCP
- scripts/ashare-retrain.sh — model_age_days 门控重训
- scripts/systemd/ashare-retrain.timer — 月度改每日
- scripts/gpu_monitor.ps1 — 双 GPU 监控
- tests/conftest.py — MagicMock 泄漏防护

### 8.3 关键设计决策

| 决策 | 理由 | 替代方案 |
|------|------|----------|
| model_age_days 驱动重训（非 IC） | live prediction 的 IC 永远是 null（数学限制） | 滞后 IC（Phase 2.2 补充） |
| Huber M-估计替换 OLS | 95% 效率，无需切模式 | OLS + 危机时切换（复杂度高） |
| PSI 监控预测分数分布 | 不需要未来标签，实时可用 | 特征级 PSI（计算量大，Phase 5 再做） |
| 跌停卖单永不取消 | 取消 = 放弃退出机制 | max_resets 后降级为市价单（A 股不支持） |
| 3 天过期硬截断 | 平衡安全与可用性 | 7 天（太宽松）、1 天（太严格） |
| flock FD 8 互斥 | 避免与 GPU lock FD 9 冲突 | systemd Conflicts=（只防 timer，不防手动） |

---

**编制完成日期**: 2026-07-15
**下次评审日期**: Gate 1 验收后（预计 2026-07-18）
