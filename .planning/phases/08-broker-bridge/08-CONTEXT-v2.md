# Phase 8 Context: Broker Bridge (v2 -- iQuant)

## 一句话

把 ashare-lab 的每日买卖信号从"纸上谈兵"变成"真金白银"。

## 架构：双机信号文件桥接

```
X500 (Linux)                         win-gpu (Windows)
  ashare-lab pipeline                  iQuant 客户端（自动登录）
  每日 18:00 生成信号                   Task Scheduler 定时启动
  -> orders/{date}.json                -> iQuant 策略读取 JSON
  -> rsync/scp 同步到 win-gpu          -> passorder() 逐笔下单
                                       -> fills/{date}.json
  <- rsync/scp 回写成交结果 <-----------
```

**为什么是这个架构**：2026年7月量化环境剧变——miniQMT 全行业关闭（7/6），华鑫个人 CTPAPI 直连已堵。剩下的路只有 QMT/iQuant 内置策略 + 信号文件桥接。

**为什么选国信 iQuant**：

| 对比项 | 国信 iQuant | 国金 QMT | 华鑫 QMT |
|--------|------------|---------|---------|
| 自动登录 | 支持 | 不支持 | 待确认 |
| 已有账户 | 是 | 是 | 仅模拟盘 |
| 佣金 | 万1可谈 | 万1 | 待确认 |
| 策略文件 | 明文GBK | 默认加密 | - |

## 实盘闸门（ADR-0001，不可跳过）

Phase 6 毕业 -> Phase 7 稳定 >=30 天 -> ADR-0003 闸门全过 -> 才能碰真钱

| # | 闸门项 | 状态 |
|---|--------|------|
| G1 | Phase 6 毕业 | 未完成 |
| G2 | Phase 7 稳定 >=30d | 未完成 |
| G3 | 符号映射 qlib->iQuant | 未完成 |
| G4 | 第一天建仓方案 | 未完成 |
| G5 | kill switch | 未完成 |
| G6 | CSRC 报备 | 未完成 |
| G7 | 模拟盘全链路 | 未完成 |
| G8 | 桥接压测 5 天 | 未完成 |

## 核心风险

| 风险 | 解决 |
|------|------|
| 符号格式不同 | qlib_to_iquant() |
| 科创板最小手 200 股 | get_lot_size() |
| iQuant 必须 Windows | 信号文件桥接 |
| 隔夜缺口 | 限价单 + 偏差 2% 跳过 |
| win-gpu 未启动 | Task Scheduler + 告警 |
| 文件同步失败 | 重试 + SHA256 + 告警 |

## 阻塞项

| # | 事项 | 状态 |
|---|------|------|
| B1 | iQuant 权限 | 已开通 |
| B2 | 佣金费率 | 待确认 |
| B3 | CSRC 报备 | 未做 |
| B4 | ADR-0003 文档 | 未做 |
| B5 | iQuant 模拟盘测试 | 未做 |

## 时间线

| 阶段 | 做什么 | 账户？ |
|------|--------|--------|
| Wave 1 | 符号映射、配置、订单导出、iQuant 策略 | 不需要 |
| Wave 2 | 信号桥接、成交导入、对账、kill switch | 模拟盘 |
| Wave 3 | 全链路测试、定时服务 | 模拟盘 |
| 实盘 | 切换生产 | G1-G8 全过 |
