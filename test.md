# 控制串口交互协议设计

本文档设计当前设备（上位机/SBC）与单片机（MCU）之间的同一个控制串口协议。该串口同时承担两类职责：

- 模式控制：MCU 向 SBC 发送模式切换命令，例如进入模式 1-4、停止、急停。
- 车体状态同步：MCU 向 SBC 发送小车状态、转向事件、圈数、停车、故障等信息，供 SBC 切换瞄准逻辑。

注意：本文档中将调试器和单片机视为同一个控制端，统一记为 MCU。联调阶段如果用串口调试器发送命令，也按 MCU 的消息格式发送，不在协议中额外区分调试器身份。

## 1. 需要收发的内容及变量名中文解释

### 1.1 最小需求

最小需求用于先把系统跑通：可以切换 4 个运行模式，可以知道小车当前边、转向、圈数、停车和故障，可以判断串口连接是否正常。

| 方向 | 消息类型 | 变量名 | 类型/取值 | 中文解释 |
| --- | --- | --- | --- | --- |
| MCU -> SBC | `MODE` | `mode_id` | `1`/`2`/`3`/`4` | 要切换到的运行模式编号。4 个控制模式分别用 1-4 表示。 |
| MCU -> SBC | `MODE` | `run_id` | 整数，可选 | 本次运行编号。每次重新开始任务时递增，方便区分旧消息。 |
| MCU -> SBC | `STOP` | `reason` | 字符串，可选 | 停止当前模式的原因，例如 `USER`、`DONE`、`TIMEOUT`。 |
| MCU -> SBC | `ESTOP` | `reason` | 字符串，可选 | 急停原因。SBC 收到后应立即停止当前模式并停止云台动作。 |
| MCU -> SBC | `EDGE` | `edge` | `AB`/`BD`/`DC`/`CA` | 小车当前所在边。赛题顺时针轨迹推荐按 `AB -> BD -> DC -> CA -> AB` 表示。 |
| MCU -> SBC | `EDGE` | `lap` | `0` 到 `5` | 当前完成圈数。刚出发可为 `0`，完成一整圈后为 `1`。 |
| MCU -> SBC | `TURN_START` | `corner` | `A`/`B`/`C`/`D` | 开始转向的顶点。 |
| MCU -> SBC | `TURN_START` | `from_edge` | `AB`/`BD`/`DC`/`CA` | 转向前所在边。 |
| MCU -> SBC | `TURN_START` | `to_edge` | `AB`/`BD`/`DC`/`CA` | 转向后将进入的边。 |
| MCU -> SBC | `TURN_START` | `lap` | `0` 到 `5` | 发生该转向事件时的完成圈数。 |
| MCU -> SBC | `TURN_END` | `corner` | `A`/`B`/`C`/`D` | 转向结束的顶点。 |
| MCU -> SBC | `TURN_END` | `edge` | `AB`/`BD`/`DC`/`CA` | 转向结束后进入的当前边。 |
| MCU -> SBC | `LAP` | `lap` | `1` 到 `5` | 已完成圈数。一般在经过 A 点并重新进入 AB 段后发送。 |
| MCU -> SBC | `PARKED` | `edge` | `AB`/`BD`/`DC`/`CA`/`TARGET` | 小车已停车到位的位置。基础要求中的固定位置可用 `TARGET`。 |
| MCU -> SBC | `FAULT` | `fault_code` | 字符串 | 故障码，例如 `LINE_LOST`、`MOTOR_STALL`、`LOW_BATTERY`。 |
| MCU <-> SBC | `PING` | `uptime_ms` | 非负整数 | 发送端从启动到当前的毫秒数，用于心跳。 |
| MCU <-> SBC | `PONG` | `uptime_ms` | 非负整数 | 对 `PING` 的回复，也可作为心跳确认。 |
| SBC -> MCU | `ACK` | `ack_seq` | 整数 | 已正确收到并处理的消息序号。 |
| SBC -> MCU | `NACK` | `ack_seq` | 整数 | 收到但无法处理的消息序号。 |
| SBC -> MCU | `NACK` | `error_code` | 字符串 | 拒绝原因，例如 `BAD_CRC`、`BAD_ARG`、`BUSY`、`UNKNOWN_TYPE`。 |

最小需求建议的工作方式：

- MCU 上电后每 1 秒发送一次 `PING` 或 `HELLO`，SBC 回复 `PONG` 或 `ACK`。
- MCU 通过按键、拨码、屏幕菜单或上级控制逻辑发送 `MODE,1` 到 `MODE,4`。
- 小车运行时，MCU 在进入每条边时发送 `EDGE`，开始转角时发送 `TURN_START`，转角结束时发送 `TURN_END`。
- 每完成一圈发送 `LAP`。
- 停车到位发送 `PARKED`，任务结束或人工停止发送 `STOP`。
- 故障时发送 `FAULT`；严重故障发送 `ESTOP`。

### 1.2 推荐需求

推荐需求用于提高瞄准连续性、联调可诊断性和抗干扰能力。

| 方向 | 消息类型 | 变量名 | 类型/取值 | 中文解释 |
| --- | --- | --- | --- | --- |
| MCU -> SBC | `HELLO` | `device_id` | 字符串 | 设备名，例如 `MSPM0_CAR`。 |
| MCU -> SBC | `HELLO` | `fw_version` | 字符串 | MCU 固件版本，例如 `v1.0.0`。 |
| MCU -> SBC | `HELLO` | `boot_ms` | 非负整数 | MCU 启动耗时或当前启动时间。 |
| SBC -> MCU | `HELLO` | `device_id` | 字符串 | SBC 设备名，例如 `RADXA_SBC`。 |
| SBC -> MCU | `STATE` | `mode_id` | `0`/`1`/`2`/`3`/`4` | SBC 当前模式。`0` 表示空闲。 |
| SBC -> MCU | `STATE` | `state` | `IDLE`/`STARTING`/`RUNNING`/`STOPPING`/`FAULT` | SBC 当前运行状态。 |
| SBC -> MCU | `STATE` | `run_id` | 整数 | 当前运行编号。 |
| MCU -> SBC | `POSE` | `edge` | `AB`/`BD`/`DC`/`CA` | 当前所在边。 |
| MCU -> SBC | `POSE` | `lap` | `0` 到 `5` | 当前完成圈数。 |
| MCU -> SBC | `POSE` | `s_mm` | 整数，单位 mm | 小车在当前边上已前进的距离。 |
| MCU -> SBC | `POSE` | `yaw_deg` | 浮点数，单位度 | 小车估计航向角。推荐 `AB=0`、`BD=90`、`DC=180`、`CA=270`。 |
| MCU -> SBC | `POSE` | `speed_mm_s` | 整数，单位 mm/s | 小车估计速度。 |
| MCU -> SBC | `POSE` | `quality` | `0` 到 `100` | 寻迹/位姿可信度，数值越大越可靠。 |
| MCU -> SBC | `TARGET` | `target_id` | 字符串/整数 | 当前目标编号。只有一个靶时可固定为 `0`。 |
| MCU -> SBC | `TARGET` | `target_state` | `VISIBLE`/`LOST`/`UNKNOWN` | MCU 侧对目标或任务状态的判断，可选。 |
| SBC -> MCU | `AIM_STATE` | `locked` | `0`/`1` | 视觉/云台是否已经锁定目标。 |
| SBC -> MCU | `AIM_STATE` | `error_px_x` | 整数 | 目标点相对画面中心的 x 方向像素误差。 |
| SBC -> MCU | `AIM_STATE` | `error_px_y` | 整数 | 目标点相对画面中心的 y 方向像素误差。 |
| SBC -> MCU | `AIM_STATE` | `stable_ms` | 非负整数 | 连续稳定瞄准的时间。 |
| MCU <-> SBC | `DIAG` | `key` | 字符串 | 诊断项名称，例如 `serial_rx_err`、`line_adc`、`cpu_temp`。 |
| MCU <-> SBC | `DIAG` | `value` | 字符串/数字 | 诊断项数值。 |
| MCU <-> SBC | `DIAG` | `level` | `INFO`/`WARN`/`ERROR` | 诊断级别。 |
| MCU <-> SBC | `TIME_SYNC` | `host_ms` | 非负整数 | SBC 当前毫秒时间，用于 MCU 对齐日志时间。 |
| MCU <-> SBC | `CONFIG` | `key` | 字符串 | 配置项名称，例如 `pose_rate_hz`、`mode_default`。 |
| MCU <-> SBC | `CONFIG` | `value` | 字符串/数字 | 配置项数值。 |

推荐发送频率：

- `PING`/`PONG`：1 Hz。
- `POSE`：10-20 Hz，足够支撑连续瞄准预测，不建议太高。
- `EDGE`、`TURN_START`、`TURN_END`、`LAP`、`PARKED`：事件发生时发送，关键事件可重复发送 2-3 次，或等待 ACK。
- `DIAG`：按需发送；普通诊断 1 Hz 以下，错误立即发送。
- `AIM_STATE`：5-10 Hz，主要给 MCU 做显示、调试或任务闭环判断。

## 2. 完整交互信息结构

### 2.1 串口参数

推荐串口参数：

```text
波特率：115200
数据位：8
校验位：None
停止位：1
流控：无
编码：ASCII/UTF-8 可兼容 ASCII
结束符：\r\n
单帧长度：建议 <= 128 字节
```

### 2.2 帧格式

推荐使用一行一帧的文本协议：

```text
$TP,<ver>,<seq>,<src>,<dst>,<type>,<payload...>*<crc>\r\n
```

字段说明：

| 字段 | 示例 | 中文解释 |
| --- | --- | --- |
| `$TP` | `$TP` | 固定帧头，表示 Twopoint Protocol。 |
| `ver` | `1` | 协议版本。当前固定为 `1`。 |
| `seq` | `42` | 消息序号，0-65535 循环递增。ACK/NACK 使用它确认消息。 |
| `src` | `MCU`/`SBC` | 发送方。调试器联调时也使用 `MCU`。 |
| `dst` | `SBC`/`MCU`/`ALL` | 接收方。点对点时通常是 `SBC` 或 `MCU`。 |
| `type` | `MODE` | 消息类型。 |
| `payload` | `2,1001` | 消息参数，按消息类型解释。 |
| `crc` | `5A` | 两位十六进制 XOR 校验，可选但推荐。 |
| `\r\n` | `\r\n` | 行结束符。 |

CRC 规则：

- 对 `$` 之后、`*` 之前的所有 ASCII 字节做 XOR。
- 结果用两位大写十六进制表示。
- 调试初期允许省略 `*<crc>`，但正式运行推荐开启。

无 CRC 的调试帧也允许：

```text
$TP,1,42,MCU,SBC,MODE,2,1001\r\n
```

正式帧示例：

```text
$TP,1,42,MCU,SBC,MODE,2,1001*27\r\n
```

下文示例中的 `*CS` 表示 CRC 占位符，实际发送时应替换为按上述规则计算出的两位十六进制校验值；联调早期也可以先省略 `*CS`。

### 2.3 消息类型定义

#### 2.3.1 模式控制

`HELLO`：设备启动或重新连接后发送。

```text
$TP,1,1,MCU,SBC,HELLO,MSPM0_CAR,v1.0.0,1200*CS\r\n
$TP,1,1,SBC,MCU,HELLO,RADXA_SBC,v1.0.0,800*CS\r\n
```

参数顺序：

```text
device_id,fw_version,boot_ms
```

`MODE`：请求 SBC 切换模式。

```text
$TP,<ver>,<seq>,MCU,SBC,MODE,<mode_id>,<run_id>*<crc>\r\n
```

示例：

```text
$TP,1,10,MCU,SBC,MODE,1,1001*CS\r\n
$TP,1,11,MCU,SBC,MODE,2,1002*CS\r\n
$TP,1,12,MCU,SBC,MODE,3,1003*CS\r\n
$TP,1,13,MCU,SBC,MODE,4,1004*CS\r\n
```

`STOP`：停止当前模式。

```text
$TP,1,20,MCU,SBC,STOP,USER*CS\r\n
$TP,1,21,MCU,SBC,STOP,DONE*CS\r\n
```

`ESTOP`：急停。

```text
$TP,1,22,MCU,SBC,ESTOP,LINE_LOST*CS\r\n
```

SBC 收到 `ESTOP` 后应立即停止当前模式，停止继续发云台运动指令，并返回状态。

`STATE`：SBC 向 MCU 回报当前运行状态。

```text
$TP,1,23,SBC,MCU,STATE,0,IDLE,1000*CS\r\n
$TP,1,24,SBC,MCU,STATE,2,STARTING,2001*CS\r\n
$TP,1,25,SBC,MCU,STATE,2,RUNNING,2001*CS\r\n
$TP,1,26,SBC,MCU,STATE,0,FAULT,2001*CS\r\n
```

参数顺序：

```text
mode_id,state,run_id
```

#### 2.3.2 车体状态与转向事件

`EDGE`：小车进入某条边。

```text
$TP,1,30,MCU,SBC,EDGE,AB,0*CS\r\n
$TP,1,31,MCU,SBC,EDGE,BD,0*CS\r\n
```

`TURN_START`：开始转向。

```text
$TP,1,40,MCU,SBC,TURN_START,B,AB,BD,0*CS\r\n
$TP,1,42,MCU,SBC,TURN_START,D,BD,DC,0*CS\r\n
$TP,1,44,MCU,SBC,TURN_START,C,DC,CA,0*CS\r\n
$TP,1,46,MCU,SBC,TURN_START,A,CA,AB,0*CS\r\n
```

`TURN_END`：转向结束。

```text
$TP,1,41,MCU,SBC,TURN_END,B,BD,0*CS\r\n
$TP,1,43,MCU,SBC,TURN_END,D,DC,0*CS\r\n
$TP,1,45,MCU,SBC,TURN_END,C,CA,0*CS\r\n
$TP,1,47,MCU,SBC,TURN_END,A,AB,1*CS\r\n
```

`LAP`：完成一圈。

```text
$TP,1,48,MCU,SBC,LAP,1*CS\r\n
```

`POSE`：小车连续位姿，推荐 10-20 Hz。

```text
$TP,1,60,MCU,SBC,POSE,AB,0,350,0.0,120,95*CS\r\n
$TP,1,61,MCU,SBC,POSE,BD,0,120,90.0,115,92*CS\r\n
```

参数顺序：

```text
edge,lap,s_mm,yaw_deg,speed_mm_s,quality
```

`PARKED`：停车到位。

```text
$TP,1,70,MCU,SBC,PARKED,TARGET,0*CS\r\n
$TP,1,71,MCU,SBC,PARKED,AB,1*CS\r\n
```

参数顺序：

```text
edge_or_position,lap
```

`AIM_STATE`：SBC 向 MCU 回报瞄准状态，方便 MCU 显示或参与任务判断。

```text
$TP,1,80,SBC,MCU,AIM_STATE,1,3,-2,500*CS\r\n
$TP,1,81,SBC,MCU,AIM_STATE,0,42,-18,0*CS\r\n
```

参数顺序：

```text
locked,error_px_x,error_px_y,stable_ms
```

#### 2.3.3 心跳与连接诊断

`PING`：心跳请求。

```text
$TP,1,100,MCU,SBC,PING,123456*CS\r\n
$TP,1,101,SBC,MCU,PING,223344*CS\r\n
```

`PONG`：心跳回复。

```text
$TP,1,102,SBC,MCU,PONG,223400,100*CS\r\n
```

参数顺序：

```text
uptime_ms,last_rx_seq
```

连接状态建议：

- 1 秒内收到对端任意有效帧：连接正常。
- 2 秒未收到有效帧：连接降级，SBC 可继续当前模式但应记录 `WARN`。
- 5 秒未收到有效帧：连接断开，SBC 应进入安全策略，例如停止当前模式或保持最后安全状态。

`DIAG`：诊断信息。

```text
$TP,1,110,MCU,SBC,DIAG,INFO,line_adc,2380*CS\r\n
$TP,1,111,MCU,SBC,DIAG,WARN,battery_mv,6900*CS\r\n
$TP,1,112,SBC,MCU,DIAG,ERROR,camera,OPEN_FAILED*CS\r\n
```

参数顺序：

```text
level,key,value
```

`FAULT`：故障。

```text
$TP,1,120,MCU,SBC,FAULT,LINE_LOST,track sensor lost line*CS\r\n
$TP,1,121,SBC,MCU,FAULT,CAMERA_LOST,camera read failed*CS\r\n
```

参数顺序：

```text
fault_code,message
```

#### 2.3.4 确认与错误

`ACK`：确认已收到并处理。

```text
$TP,1,200,SBC,MCU,ACK,10,MODE*CS\r\n
```

参数顺序：

```text
ack_seq,ack_type
```

`NACK`：拒绝或解析失败。

```text
$TP,1,201,SBC,MCU,NACK,10,BUSY,current mode is stopping*CS\r\n
$TP,1,202,SBC,MCU,NACK,12,BAD_ARG,unknown mode_id*CS\r\n
$TP,1,203,SBC,MCU,NACK,15,BAD_CRC,crc mismatch*CS\r\n
```

参数顺序：

```text
ack_seq,error_code,message
```

需要 ACK 的消息：

- `MODE`
- `STOP`
- `ESTOP`
- `TURN_START`
- `TURN_END`
- `LAP`
- `PARKED`
- `FAULT`

可以不 ACK 的消息：

- `POSE`
- 高频 `DIAG`
- 周期性 `PING`/`PONG`

### 2.4 典型交互示例

#### 示例 A：基础模式，停车后瞄准

```text
MCU -> SBC: $TP,1,1,MCU,SBC,HELLO,MSPM0_CAR,v1.0.0,1200*CS
SBC -> MCU: $TP,1,1,SBC,MCU,HELLO,RADXA_SBC,v1.0.0,800*CS
MCU -> SBC: $TP,1,2,MCU,SBC,MODE,2,2001*CS
SBC -> MCU: $TP,1,2,SBC,MCU,ACK,2,MODE*CS
SBC -> MCU: $TP,1,3,SBC,MCU,STATE,2,STARTING,2001*CS
SBC -> MCU: $TP,1,4,SBC,MCU,STATE,2,RUNNING,2001*CS
MCU -> SBC: $TP,1,3,MCU,SBC,PARKED,TARGET,0*CS
SBC -> MCU: $TP,1,5,SBC,MCU,ACK,3,PARKED*CS
SBC -> MCU: $TP,1,6,SBC,MCU,AIM_STATE,1,3,-2,500*CS
MCU -> SBC: $TP,1,4,MCU,SBC,STOP,DONE*CS
SBC -> MCU: $TP,1,7,SBC,MCU,STATE,0,IDLE,2001*CS
```

#### 示例 B：发挥模式，小车边走边瞄准一圈

```text
MCU -> SBC: $TP,1,10,MCU,SBC,MODE,4,3001*CS
SBC -> MCU: $TP,1,10,SBC,MCU,ACK,10,MODE*CS
SBC -> MCU: $TP,1,11,SBC,MCU,STATE,4,RUNNING,3001*CS

MCU -> SBC: $TP,1,11,MCU,SBC,EDGE,AB,0*CS
MCU -> SBC: $TP,1,12,MCU,SBC,POSE,AB,0,100,0.0,120,96*CS
MCU -> SBC: $TP,1,13,MCU,SBC,POSE,AB,0,500,0.0,120,95*CS

MCU -> SBC: $TP,1,14,MCU,SBC,TURN_START,B,AB,BD,0*CS
SBC -> MCU: $TP,1,12,SBC,MCU,ACK,14,TURN_START*CS
MCU -> SBC: $TP,1,15,MCU,SBC,TURN_END,B,BD,0*CS
SBC -> MCU: $TP,1,13,SBC,MCU,ACK,15,TURN_END*CS

MCU -> SBC: $TP,1,16,MCU,SBC,EDGE,BD,0*CS
MCU -> SBC: $TP,1,17,MCU,SBC,POSE,BD,0,250,90.0,115,93*CS

MCU -> SBC: $TP,1,18,MCU,SBC,TURN_START,D,BD,DC,0*CS
MCU -> SBC: $TP,1,19,MCU,SBC,TURN_END,D,DC,0*CS
MCU -> SBC: $TP,1,20,MCU,SBC,EDGE,DC,0*CS

MCU -> SBC: $TP,1,21,MCU,SBC,TURN_START,C,DC,CA,0*CS
MCU -> SBC: $TP,1,22,MCU,SBC,TURN_END,C,CA,0*CS
MCU -> SBC: $TP,1,23,MCU,SBC,EDGE,CA,0*CS

MCU -> SBC: $TP,1,24,MCU,SBC,TURN_START,A,CA,AB,0*CS
MCU -> SBC: $TP,1,25,MCU,SBC,TURN_END,A,AB,1*CS
MCU -> SBC: $TP,1,26,MCU,SBC,LAP,1*CS
SBC -> MCU: $TP,1,14,SBC,MCU,ACK,26,LAP*CS

MCU -> SBC: $TP,1,27,MCU,SBC,STOP,DONE*CS
SBC -> MCU: $TP,1,15,SBC,MCU,STATE,0,IDLE,3001*CS
```

#### 示例 C：连接心跳与诊断

```text
MCU -> SBC: $TP,1,100,MCU,SBC,PING,100000*CS
SBC -> MCU: $TP,1,100,SBC,MCU,PONG,99000,100*CS
SBC -> MCU: $TP,1,101,SBC,MCU,DIAG,INFO,mode,4*CS
MCU -> SBC: $TP,1,101,MCU,SBC,DIAG,WARN,battery_mv,6900*CS
```

#### 示例 D：故障和急停

```text
MCU -> SBC: $TP,1,130,MCU,SBC,FAULT,LINE_LOST,track lost for 200ms*CS
SBC -> MCU: $TP,1,130,SBC,MCU,ACK,130,FAULT*CS
MCU -> SBC: $TP,1,131,MCU,SBC,ESTOP,LINE_LOST*CS
SBC -> MCU: $TP,1,131,SBC,MCU,ACK,131,ESTOP*CS
SBC -> MCU: $TP,1,132,SBC,MCU,STATE,0,FAULT,3001*CS
```

### 2.5 SBC 侧处理建议

SBC 收到消息后的基本策略：

```text
MODE:
  如果当前空闲，启动对应模式程序。
  如果已有模式运行，先停止当前模式，再启动新模式。
  回复 ACK；如果模式不存在或正在急停，回复 NACK。

STOP:
  停止当前模式，释放摄像头和电机串口，回复 ACK。

ESTOP:
  立即停止当前模式和云台动作，回复 ACK，并进入 FAULT 状态。

EDGE / TURN_START / TURN_END / LAP / POSE:
  更新车体状态缓存，供当前模式读取。
  关键事件回复 ACK；POSE 不必逐帧 ACK。

PING:
  回复 PONG，并更新连接时间。

FAULT:
  记录故障。严重故障可主动停止模式。
```

推荐 SBC 内部维护这些状态变量：

| 变量名 | 中文解释 |
| --- | --- |
| `current_mode_id` | 当前正在运行的模式编号，空闲为 `0`。 |
| `current_run_id` | 当前任务运行编号。 |
| `runtime_state` | 当前状态：空闲、启动中、运行中、停止中、故障。 |
| `vehicle_edge` | 小车当前所在边。 |
| `vehicle_lap` | 当前完成圈数。 |
| `vehicle_corner` | 最近一次转向对应的角点。 |
| `vehicle_turning` | 当前是否正在转向。 |
| `vehicle_s_mm` | 小车在当前边上的估计前进距离。 |
| `vehicle_yaw_deg` | 小车估计航向角。 |
| `vehicle_speed_mm_s` | 小车估计速度。 |
| `vehicle_quality` | 小车位置/寻迹质量。 |
| `last_rx_monotonic` | 最近收到有效串口帧的本机时间，用于连接超时判断。 |
| `last_mcu_seq` | 最近收到的 MCU 消息序号。 |
| `serial_link_state` | 串口连接状态：正常、降级、断开。 |
