# 摄像头实时换衣 · 最小可演示原型（CPU）

打开摄像头，实时检测人体上半身，把 `assets/clothes/` 里的**透明 PNG 上衣**按肩宽缩放、
按肩线角度旋转，锚点对齐双肩中点后叠到画面上。

衣柜共 **15 件**：3 件代码生成的占位上衣 + **12 件从你上传的穿搭拼图里切出来的真实上衣**。
按 `n` / `p` 前后翻遍整个衣柜，`1/2/3` 快速切前三件，`0` 隐藏，`h` 开关底部缩略图条，
`q` 退出，窗口显示 FPS。

> 这是**原型**：只保证正面站立时"大致贴合"，不做遮挡、褶皱、形体变形、多人、侧面/背面。

---

## 1. 资源目录结构（衣服图片分类归档）

```
assets/
├── clothes/                         # 所有"衣服"资源
│   ├── placeholder/                 # 组 1：代码生成的 3 件占位上衣（无真实素材时的兜底）
│   │   ├── jacket_black.png         #   447x367 BGRA
│   │   ├── jacket_red.png           #   447x367 BGRA
│   │   └── shirt_white.png          #   447x367 BGRA
│   └── integrated/                  # 组 2：从上传拼图切出的 12 件真实上衣（700px 宽 BGRA）
│       ├── look_01.png ~ look_12.png
├── refs/                            # 你上传的原始参考图（只归档，不参与渲染）
│   ├── ref_person.jpg               #   原始人像（白衣模特全身正面）
│   └── ref_looks_collage.jpg        #   原始 4x3 穿搭平铺拼图（12 套）
├── uploads/                         # 上传样图模块的临时目录（中间产物，可随时删）
│   └── tmp/<session>/               #   每次上传一个 session：原图副本 + upload.json + result.json + items/*.png
└── pose_landmarker_lite.task        # MediaPipe Pose 模型（约 5.8MB）
```

**分类原则**：`clothes/` 只放"能叠到人身上的上衣"，按"来源/可信度"分成两组；
`refs/` 放原始素材（不是渲染素材，但保留可追溯性）。
衣柜启动时**自动扫描** `assets/clothes/`，无需手工登记——往 `integrated/` 里丢一张
合规 PNG，重启（或按 `r`）就会多出一件衣服。

### 12 件真实上衣怎么来的

`assets/refs/ref_looks_collage.jpg` 是一张 **4 列 x 3 行 = 12 格**的整套穿搭平铺拼图，
每格是"上衣 + 阔腿裤 + 配饰"的完整一套。本项目只叠上半身，所以做了这些处理：

1. 用"整行/整列近白 band"定位出 4x3 的内容区，得到每格的精确边界；
2. 逐格 `floodFill（从四边漫水 + FIXED_RANGE）` 去白底，再用"灰度 < 205 强制实心"兜底
   （**不能**用全局白阈值，否则浅色衣物会被挖出洞）；
3. 按人工量出的"上衣下摆"比例把下半身（裤子）横切掉；
4. 裁到 alpha 非零的 bbox，等比放大到宽 700（`INTER_LANCZOS4`），存成 BGRA PNG。

切图脚本是一次性工具，**不是交付物**；成品就是 `assets/clothes/integrated/look_*.png`。
每件上衣的"肩线在 PNG 高度上的比例"量好后写进了 `CONFIG["anchor_y"]`。

## 2. 本机实测环境

| 项 | 版本 |
| --- | --- |
| Python | 3.14.6（`C:\Users\HP\AppData\Local\Programs\Python\Python314`） |
| 虚拟环境 | `c:\Users\HP\Desktop\TRAE_work\.venv` |
| mediapipe | 1.1.1（**Tasks API**，`mp.solutions` 已在 1.x 移除） |
| opencv-contrib-python | 5.0.0.93 |
| numpy | 2.5.4 |
| 摄像头 | index 0 可用，默认采集 1280x720 |

纯 CPU 运行，不需要 GPU，不训练模型。实测单帧耗时：姿态检测 ≈13ms + 叠加 ≈7ms ≈ **50 FPS**
（960x540 满屏基准，不含摄像头取帧），实机通常 25~40 FPS。

### 环境要求（最低要求）

| 项 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Windows 10 / 11 | 摄像头枚举走 DirectShow / MSMF 后端。非 Windows 仍可用 `--image` 静态图模式验证算法链路 |
| Python | **3.10 及以上** | 实测 3.14.6；mediapipe 1.x 要求 3.9+，代码用现代类型标注（`str \| None`）故建议 3.10+ |
| 第三方依赖 | 见 [requirements.txt](file:///c:/Users/HP/Desktop/TRAE_work/requirements.txt) | `mediapipe` / `opencv-contrib-python` / `numpy`，全部可用 pip 安装 |
| 硬件 | 普通 x86 CPU | 无需 GPU；实时换衣需要摄像头（USB 或笔记本内置） |
| 图形界面 | 需要能弹出窗口的桌面环境 | 实时预览用 OpenCV `imshow`；上传样图模块的"选择文件"对话框需要 `tkinter`（Windows 官方 Python 自带，无需额外安装） |
| 磁盘 | 约 100 MB | 依赖装在 `.venv` 里；姿态模型约 5.8 MB（本仓库已附带） |
| 网络 | 一般不需要 | 仅当 `assets/pose_landmarker_lite.task` 缺失时才需联网下载 |

> 只要满足上表，按第 3 节装依赖、第 4 节跑命令即可。没有摄像头也能用
> `python main.py --image 照片.jpg --out preview.jpg` 和 `python main.py --selftest` 完整验证。

## 3. 安装

```powershell
cd c:\Users\HP\Desktop\TRAE_work
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 首次装依赖如走 pypi.org 会很慢甚至卡住，**建议加上面的清华镜像**。

首次运行 `main.py` 时会自动下载姿态模型 `assets/pose_landmarker_lite.task`（约 5.8MB，
本仓库已附带，无需再下）。若无法访问 Google 存储，可改装自带模型的老版本：
`pip install "mediapipe==0.10.35"`（该版本仍有 `mp.solutions` API，代码会自动兼容）。

## 4. 运行

```powershell
cd c:\Users\HP\Desktop\TRAE_work
.\.venv\Scripts\python.exe main.py
```

命令行参数：

| 命令 | 作用 |
| --- | --- |
| `python main.py` | 打开摄像头 index 0，窗口 960x540 |
| `python main.py --camera 1` | 换摄像头索引 |
| `python main.py --size 1280x720` | 自定义窗口分辨率（也支持 `1280,720` / `1280*720`） |
| `python main.py --no-hud` | 关闭底部缩略图条 |
| `python main.py --probe` | 只探测 0~3 哪个索引可用（排障用） |
| `python main.py --debug` | 启动即画骨架 + 鼻/肩/髋关键点 + 双肩中点十字线（调参用） |
| `python main.py --image 照片.jpg --out preview.jpg` | **没有摄像头也能验证**：对静态图跑完整流程并存结果 |
| `python main.py --image 照片.jpg --clothes 8` | 静态图 / 摄像头模式指定用衣柜里第几件（0 起，越界自动回退到 0） |
| `python main.py --selftest` | **全面自检**：逐件校验 + 渲染 + contact sheet + 缓存/耗时，见第 7 节 |
| `python main.py --upload 样图.jpg` | **上传样图并自动提取衣物**，打印标准化 JSON（`garment.v1`）后退出，见第 13 节 |
| `python main.py --upload 样图.jpg --no-upload-preview` | 同上但跳过上传前预览窗口（无 GUI / 批量场景） |
| `python main.py --upload 样图.jpg --upload-add` | 提取到的衣物**注入衣柜内存**后继续进入摄像头模式（按 `n/p` 翻到「上传-xx」即可穿上） |
| `python main.py --upload 样图.jpg --upload-backend ai_stub` | 指定提取后端（默认 `white_bg`；`ai_stub` 是 AI 模型对接桩） |

按键：

| 键 | 作用 |
| --- | --- |
| `1` / `2` / `3` | 快速切到衣柜第 1/2/3 件（即三件占位上衣） |
| `n` / `p` | 衣柜内前后切换（循环 15 件） |
| `0` | 隐藏衣服 |
| `r` | 重新扫描 / 校验衣柜（已缓存且未改动的文件不会重复解码） |
| `h` | 开关底部缩略图条（HUD） |
| `d` | 开关骨架 + 锚点调试画线 |
| `u` | **上传一张样图 → 自动提取衣物 → 追加进衣柜（仅内存）**，随后按 `n/p` 即可穿上 |
| `q` / `ESC` | 退出 |

启动时会打印一张**加载校验表**，逐件标注 `PASS/FAIL` 与原因（尺寸、是否有 alpha、
不透明像素占比等），FAIL 的不会进入衣柜。

## 5. 图片加载 / 渲染 / 缓存（要求 2、4 的实现）

> 本项目是 **Python + OpenCV 桌面原型，没有 Web UI**。所以"UI 布局 / 多设备 /
> 浏览器图片缓存"这类 Web 语境的要求，全部落成**等价且可验证**的桌面实现（见下）。

**加载与渲染**（`main.py` 的"3. 衣柜"与"4. 透明 PNG 叠加"两节）：

- **加载**：`_imread_unicode()` 用 `cv2.imdecode` 读图，兼容中文路径；逐件校验
  "有 4 通道 / 尺寸 ≥32 / 不透明像素占比 ≥2%"。
- **不失真**：`overlay_item()` 做**等比缩放**（`sx = sy`，由图宽算比例），
  旋转用 `warpAffine` 的旋转矩阵——等比 + 刚体旋转，所以图不会被拉扁或拉伸。
- **无黑边**：先用 `src_rgb = rgb/255 * alpha`（预乘 alpha）再插值，
  缩放/旋转后的半透明边缘不会渗出黑边。
- **锚点对齐**：PNG 内的"肩线锚点"（`anchor`）映射到双肩中点，再按肩线倾角旋转。

**缓存机制**（`class Wardrobe`）：按**绝对路径 + mtime** 做进程内缓存，
每个文件只在首次加载时解码一次，并预先算好三样东西——
解码后的 `src_a`（float32 透明通道）、`src_rgb`（预乘色）、`thumb`（HUD 缩略图）。
逐帧渲染直接用缓存好的数组，**不重算 `alpha/255` 与预乘**；`reload()` 会复用缓存，
只有 mtime 变了才重新解码。`--selftest` 里会实测这一点（"缓存命中"那一行）。

实测：缓存后单件叠加 **≈0.95 ms**（40 轮 x 15 件平均，960x540）。

## 6. 不同屏幕尺寸 / 设备的展示一致性（要求 3）

因为没有 Web 布局，一致性靠"**全部用相对量**"来保证：

- **衣服尺寸相对肩宽**：衣服宽度 = 检测到的肩宽(px) × `scale`。人离镜头远近、
  摄像头分辨率、窗口大小怎么变，衣服都会跟着人一起缩放，比例恒定。
- **HUD 相对窗口宽**：底部缩略图条的格子尺寸、间距、字号全按 `frame` 宽度算，
  所以 960x540 和 1280x720 下观感一致；窗口宽度可配（`--size`）。
- **显示分辨率可配**：`--size WxH` 直接覆盖 `CONFIG["view_width"]/view_height`，
  采集端统一 resize 到该分辨率，算法全部在该分辨率下运行。

## 7. 全面自检（要求 5）

```powershell
.\.venv\Scripts\python.exe main.py --selftest
```

会依次做 5 步，并输出一张 `selftest_contact_sheet.jpg`（每件一张"叠加后"的缩略格）：

1. **加载与校验**：打印 15 件的 `PASS/FAIL` 表；
2. **准备输入帧**：优先用 `assets/refs/ref_person.jpg` 真实检出双肩；检不出就退化为
   合成"人形"参考帧（并明确标注用的是几何约束）；
3. **逐件渲染**：把每件叠上去，统计"真正被改动的像素数"来判定是否真的画出来了
   （0 像素 = 显示异常）；
4. **缓存命中 + 耗时**：二次 `reload()` 后缓存条目数不变 = 未重复解码；再测单件耗时；
5. **落盘 contact sheet** 供肉眼复核。

本机实测结果：**15 件全部 PASS**，加载失败 0、渲染失败 0、缓存命中、单件 0.95 ms。
若有任何 FAIL，进程退出码为 1（方便接 CI）。

## 8. 演示流程（建议照此演示）

1. `.\\.venv\\Scripts\\python.exe main.py --debug` —— 先让观众看到黄点(鼻)/橙点(双肩)/蓝点(双髋)
   和绿色十字(双肩中点)，证明姿态关键点是实时的。
2. 按 `d` 关掉调试线，正对镜头站好（上半身入画、光线均匀、背景干净）。
3. 按 `1`、`2`、`3` 切换占位上衣；按 `n` 翻到第 4 件开始就是**上传拼图切出的真实上衣**，
   连按 `n` 一路翻完 12 件，说明"衣服宽度 = 肩宽 × scale，旋转 = 肩线倾角"。
4. 按 `0` 隐藏，再按 `n` 切回来；左右歪头/耸肩，看衣服跟着肩线转。
5. 按 `h` 开关底部缩略图条。
6. 走到画面外或背对镜头 —— 程序不崩溃，顶部状态栏提示"未检测到双肩"并停止叠加。
7. 按 `q` 退出。

## 9. 调参（只改 main.py 顶部的 `CONFIG`）

```python
"clothes": [                        # 占位组（写死在 CONFIG 里的 3 件）
    {
        "name": "黑色外套",
        "path": "assets/clothes/placeholder/jacket_black.png",
        "scale": 1.87,         # 衣服宽度 = 肩宽 * scale
        "offset_x": 0.00,      # 水平偏移，单位 = 肩宽（正 = 向右）
        "offset_y": -0.05,     # 垂直偏移，单位 = 肩宽（正 = 向下）
        "angle_offset": 0.0,   # 叠加到肩线倾角上的额外角度（度）
        "alpha": 0.95,         # 不透明度
        "anchor": (0.5, 0.0),  # PNG 内的"肩线锚点"（归一化 0~1），该点对齐双肩中点
    },
    ...
],

# 集成组（integrated/）统一用这套默认值，再叠加每件自己的 anchor_y
"integrated_default": {"scale": 1.65, "offset_x": 0.0, "offset_y": -0.03,
                       "angle_offset": 0.0, "alpha": 0.95},
"anchor_y": {"look_01.png": 0.133, ...},   # 每件"肩线在 PNG 高度上的比例"
"tune": {},                                # 想单独微调某件就写在这里（按文件名）
```

**anchor_y 是什么**：`integrated/` 里的图是从拼图切出来的，每件上衣在 PNG 里顶部的
留白不同，所以"肩线在高度上的比例"要逐件量（`anchor_y` 那张表就是量出来的结果）。
`tune` 可以给某一件单独改 `scale`/`offset_y`，例如
`"tune": {"look_07.png": {"scale": 1.8, "offset_y": -0.06}}`。

**调参顺序**：先 `anchor`（让衣服肩线对准双肩中点）→ 再 `scale`（对准躯干宽度）
→ 最后 `offset_x/offset_y`（微调）。开 `--debug` 会画出双肩中点十字线，对着看最直观。

### 加一张自己的衣服图

1. 做一张**背景透明**的上衣 PNG（GIMP/PS/抠图工具都行），正面上身、对称、袖子顺着胳膊往下垂；
2. 丢进 `assets/clothes/integrated/`（或 `placeholder/`）；
3. 重启（或按 `r`）—— 衣柜会自动收录并在启动表里给出 `PASS/FAIL`；
4. 若位置偏了，在 `CONFIG["tune"]` 里按文件名微调。

### 占位衣服

`assets/clothes/placeholder/` 下没有对应 PNG 时，`main.py` 会**自动用 OpenCV 代码生成**
三张占位透明上衣（穿着态剪影、自动裁掉透明边），保证没有真实素材也能立刻跑起来。

## 10. 已知限制

- 只对**正面、单人、上半身**有效；侧面、背面、坐姿、蹲姿会明显错位。
- 衣服是**平面刚性贴图**：不跟随身体形变，转身/抬手时袖子穿帮、衣服不"包裹"身体属正常现象。
- 不做遮挡处理：手、头发、随身物体与衣服的前后关系会出错。
- 肩宽只取左右肩关键点（MediaPipe 的肩关节位置），比肉眼看到的肩宽略窄，所以 `scale` 会偏大。
- 从拼图切出的 12 件上衣里，`look_02` / `look_03` 被切得偏短（原图下摆本就高），属可接受的瑕疵。
- 双肩可见性 < 0.5 时**保持上一帧位置**、不叠加错位；连续 8 帧以上检测不到会在画面下方提示。
- 关键点稳定性依赖光照（均匀）、背景（干净）、人物占比（上半身尽量占满画面 1/3 以上）。
- 性能：如卡顿可在 `CONFIG` 把 `model_complexity` 改为 `0`，或把 `view_width/view_height` 调小。

## 11. 常见问题

**摄像头打不开**（程序会打印排错清单）：

1. 摄像头被别的程序占用（微信 / 钉钉 / 腾讯会议 / 浏览器网页会议），全部关掉；
2. `Windows 设置 → 隐私和安全性 → 相机`：打开"相机访问"和"让桌面应用访问相机"；
3. 笔记本是否有物理摄像头开关或遮蔽盖；
4. `python main.py --camera 1` 换索引；先 `python main.py --probe` 看哪个能用；
5. 没有摄像头也能验证算法链路：`python main.py --image 你的照片.jpg --out preview.jpg`。

**启动表里有 FAIL**：表里会写明原因（不是 PNG / 没有 alpha / 尺寸过小 / 几乎全透明）。
按原因修好文件后按 `r` 重新扫描即可，不用重启程序。

**提示缺 `pose_landmarker_lite.task` 且下载失败**：
手动下载
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`
放到 `assets/`；或 `pip install "mediapipe==0.10.35"`。

**`can't open file` / 读不到图片**：`--image` 路径含中文也支持（代码用 `imdecode` 兜底），
但请确认扩展名是常见图片格式且文件真实存在。

**衣服位置偏了**：见第 9 节调参顺序，配合 `--debug` 十字线。

## 12. 本次集成的关键说明

- 你上传的**两张 JPG**都不是透明 PNG：一张是人像、一张是 4x3 穿搭平铺拼图。
  处理方式是：人像归档进 `assets/refs/`（并作为自检的输入帧），拼图**切成 12 件上衣**
  进 `assets/clothes/integrated/`；另外保留 3 件代码生成的占位上衣兜底。
- 每次启动都会**自动扫描 + 校验**衣柜并打印 `PASS/FAIL` 表；`--selftest` 提供
  "全部可访问、无加载失败、无显示异常"的可验证证据（contact sheet + 汇总）。
- 集成**不影响原有行为**：`1/2/3` 换衣、`0` 隐藏、`q` 退出、`d` 调试全部保留，
  只是衣柜从 3 件扩到 15 件并新增 `n/p`、`r`、`h` 三个附加按键。
- 本机只有 Python 3.14，而 mediapipe 1.x 已移除 `mp.solutions`，故改用官方 Tasks API
  并自动下载 `.task` 模型；代码同时保留对 `mediapipe==0.10.35` 的兼容分支。
- 只按需求实现 CPU 单文件原型，未接 LLM / 语音 / 生成式换衣 / Web UI。

## 13. 用户上传样图模块（上传 + 衣物元素自动提取）

独立模块 [garment_upload.py](file:///c:/Users/HP/Desktop/TRAE_work/garment_upload.py)，
**接口文档见 [docs/UPLOAD_MODULE.md](file:///c:/Users/HP/Desktop/TRAE_work/docs/UPLOAD_MODULE.md)**
（含配置项表、错误码表、数据结构字段表、AI 模型接入指南、扩展点速查）。

> 同样地，本项目没有 Web UI，所以"上传控件 / 上传前预览"落成**桌面等价实现**：
> 系统文件选择对话框 = 上传入口，OpenCV 窗口 = 上传前预览，控制台文本 = 错误提示。

它提供 6 件事：

1. **上传交互入口**：`pick_file_dialog()` 选文件；`preview_image()` 上传前预览（Enter/Space 确认，
   ESC 取消）；`validate_image()` 做格式（JPG/PNG/WEBP/BMP）/ 大小（≤12MB）/ 尺寸 / 可解码校验。
2. **衣物元素自动提取**：`BaseExtractor` 标准接口 + 注册表。默认后端 `white_bg`（白底抠图，纯 CPU 开箱即用）；
   `AiExtractorStub` 是**给 AI 模型预留的对接桩**，内部三段式钩子 `_detect`（检测）/`_segment`（分割）/`_classify`（分类）
   带 TODO，接 YOLO+SAM+CLIP 这类模型时**只改这三个方法**，主程序与下游代码零改动。
3. **标准化衣物数据结构**：`GarmentItem`（类别 / `bbox` 位置坐标 / `polygon` 轮廓信息 / 掩码与透明抠图 PNG /
   属性）与 `GarmentResult`，`to_dict()` 输出带 `schema_version: "garment.v1"` 的 JSON，
   可直接对接后续**衣物编辑 / 替换 / 搭配**业务。
4. **临时存储 + 异常处理**：`TempStore` 按 session 目录存"原图副本 + meta.json + 每件的 mask/rgba PNG"，
   支持过期清理；异常统一为 `UploadError → ValidationError / ExtractionError`（带 `error_code`），
   提取失败按 `retries` 自动重试，重试耗尽返回结构化错误供上层提示。
5. **功能开关与参数配置**：`UploadConfig`（`enabled` 总开关、`precision` 精度档位 fast/balanced/high、
   `allowed_categories` 支持类型范围、`extractor` 后端选择、`extractor_params` 逐项覆盖）。
6. **独立可测试**：见下方自检命令。

```powershell
# 模块独立自检（自己合成样图跑完整流水线 + 4 类失败用例 + 临时存储/清理），退出码 0/1
.\.venv\Scripts\python.exe garment_upload.py --self-test

# 对真实图片提取衣物并打印标准化 JSON
.\.venv\Scripts\python.exe main.py --upload assets\refs\ref_looks_collage.jpg --no-upload-preview

# 列出可用提取后端 / 清理过期临时 session
.\.venv\Scripts\python.exe garment_upload.py --list-backends
.\.venv\Scripts\python.exe garment_upload.py --cleanup
```

**与主程序的"无缝对接"**：`Wardrobe.add_slot_from_bgra()` 把提取到的抠图按**与静态资源完全相同的
校验与缓存字段**装进 `wardrobe.slots`，于是 `overlay_item()` 能像对待普通衣柜条目一样直接渲染它。
摄像头里按 `u` 上传、或 `--upload 样图.jpg --upload-add`，都能做到"提取完立刻穿上"。

实测（`assets/refs/ref_looks_collage.jpg`）：检出 8 件 → 全部注入衣柜 → 衣柜 15 → 23 件 → 8 件均可正常渲染。
自检证据图：`upload_selftest_sheet.jpg`。

**不影响原有功能**：模块独立成文件，`main.py` 里是懒加载挂载；`main.py --selftest` 仍是
15 件全 PASS、退出码 0，`1/2/3`、`0`、`q`、`d` 等原按键全部保留，只新增 `u` 与 `--upload*` 参数。
