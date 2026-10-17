# 用户上传样图模块 —— 接口文档

模块文件：[garment_upload.py](file:///c:/Users/HP/Desktop/TRAE_work/garment_upload.py)（约 1200 行，中文注释）

> 这是"用户上传样图 → 自动提取衣物 → 交给下游编辑/替换/搭配"这条链路的**预留功能模块**。
> 它只依赖 **标准库 + opencv(cv2) + numpy**，不 import `main.py`，**可以独立运行与独立测试**。

## 0. 一句话定位

| 能力 | 实现位置 | 可替换 / 可扩展 |
| --- | --- | --- |
| 上传入口（选文件） | `pick_file_dialog()` | 可换成拖拽 / Web 上传控件 |
| 上传前预览 | `preview_image()` | Enter/Space 确认，ESC 取消 |
| 上传校验 | `validate_image()` | 格式 / 大小 / 尺寸 / 可解码 |
| 衣物元素提取 | `BaseExtractor` + `WhiteBgExtractor` / `AiExtractorStub` | **接 AI 模型只需实现 3 个钩子** |
| 标准化数据契约 | `GarmentItem` / `GarmentResult` | 字段即下游对接契约，可加字段 |
| 临时存储 | `TempStore` | session 目录 + 过期清理 |
| 异常与重试 | `UploadError` / `ValidationError` / `ExtractionError` | 错误码表见 §3 |
| 开关与参数 | `UploadConfig` | 精度档位 / 类型范围 / 后端选择 |

**桌面等价物说明**：本项目是 Python + OpenCV 桌面原型，没有 Web UI。
所以"上传控件 / 上传前预览"在本模块里落成：**系统文件选择对话框 = 上传入口**、
**OpenCV 窗口 = 上传前预览**、**控制台文本 = 错误提示**，语义与 Web 版一一对应且可验证。

---

## 1. 快速开始

```powershell
# 独立自检（不依赖任何外部素材，自己合成一张白底样图跑完整流水线）
.\\.venv\\Scripts\\python.exe garment_upload.py --self-test

# 对真实图片提取衣物并打印标准化 JSON（同时落一张可视化证据图）
.\\.venv\\Scripts\\python.exe garment_upload.py --image 样图.jpg

# 看有哪些提取后端；清理过期的临时 session
.\\.venv\\Scripts\\python.exe garment_upload.py --list-backends
.\\.venv\\Scripts\\python.exe garment_upload.py --cleanup
```

Python 侧最小用法：

```python
from garment_upload import UploadConfig, upload_and_extract, load_item_bgra

cfg = UploadConfig(precision="balanced", extractor="white_bg")
uploaded, result = upload_and_extract("样图.jpg", cfg)   # 上传失败会抛 UploadError
if result.ok:
    for item in result.items:
        print(item.category.value, item.bbox.to_dict(), item.rgba_png)
        bgra = load_item_bgra(item)                      # 拿去编辑 / 替换 / 搭配
else:
    print(result.error["error_code"], result.error["error"])
```

---

## 2. 配置项 `UploadConfig`（要求 5）

全部开关集中在这一个 dataclass，改参数不用改代码逻辑。`UploadConfig().to_dict()` 可导出为 JSON。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `True` | **总开关**。`False` 时所有上传入口直接报 `E_DISABLED` |
| `allowed_formats` | `("jpg","jpeg","png","webp","bmp")` | 允许的扩展名（小写、不含点） |
| `max_file_mb` | `12.0` | 单文件大小上限 |
| `min_side_px` / `max_side_px` | `64` / `8000` | 最小边下限 / 最大边上限 |
| `tmp_dir` | `assets/uploads` | 临时存储根目录 |
| `session_ttl_hours` | `24.0` | session 保留时长，超时被 `cleanup()` 回收 |
| `preview` | `True` | 是否启用上传前预览确认 |
| `retries` | `1` | **提取失败自动重试次数**（0=不重试） |
| `precision` | `"balanced"` | **识别精度档位** `fast` / `balanced` / `high` |
| `allowed_categories` | 全部 6 类 | **支持的衣物类型范围**（只留上衣就填 `("top","outerwear")`） |
| `extractor` | `"white_bg"` | 提取后端名，见 `list_extractors()` |
| `extractor_params` | `{}` | 透传给后端，**逐项覆盖**精度档位里的同名键 |

### 识别精度档位 `PRECISION_PRESETS`

| 档位 | 处理长边 | 抠图容差 `tol` | 强制实心阈值 `white_thr` | 轮廓简化 `eps` | 最小面积占比 `min_area` |
| --- | --- | --- | --- | --- | --- |
| `fast` | 640 | 22 | 205 | 0.020 | 0.020 |
| `balanced` | 1280 | 14 | 205 | 0.010 | 0.012 |
| `high` | 2200 | 9 | 208 | 0.005 | 0.008 |

想微调某一项而不动档位，用 `extractor_params`，例如：
`UploadConfig(extractor_params={"min_area": 0.005, "max_items": 20})`。

---

## 3. 异常与错误码（要求 4）

异常层级：`Exception` → `UploadError` → `ValidationError` / `ExtractionError`。
每个异常都有 `.code`（错误码）、`.message`、`.detail`，以及 `.to_dict()`（能直接塞进 JSON 响应）。

| 错误码 | 抛出点 | 含义 / 上层应对 |
| --- | --- | --- |
| `E_DISABLED` | 校验 / 提取 | 功能被 `enabled=False` 关闭，别弹重试 |
| `E_CANCELLED` | 选文件 / 预览 | **用户主动取消**，静默处理即可 |
| `E_NOT_FOUND` | 校验 | 文件不存在 |
| `E_NOT_FILE` | 校验 | 路径不是普通文件 |
| `E_FORMAT` | 校验 | 扩展名不在白名单 → 提示"仅支持 JPG/PNG/WEBP/BMP" |
| `E_EMPTY` | 校验 | 0 字节文件 |
| `E_SIZE` | 校验 | 超过 `max_file_mb` → 提示压缩后重试 |
| `E_DECODE` | 校验 / 预览 | 不是有效图片或已损坏 → 提示换一张 |
| `E_DIMENSION` | 校验 | 超出 `min_side_px` / `max_side_px` |
| `E_TMP_WRITE` | 临时存储 | 写临时目录失败 → 检查磁盘权限/空间 |
| `E_BACKEND` | 提取 | 后端名未注册 → 提示可用后端列表 |
| `E_BACKEND_UNAVAILABLE` | 提取 | 后端不可用（AI 桩未接模型）→ 提示去实现钩子 |
| `E_NO_GARMENT` | 提取 | 没识别到衣物元素 → 提示换背景干净的图 |
| `E_EXTRACT` | 提取 | 后端内部异常，**已按 `retries` 重试完** → 可提示"重试" |

**重试机制**：`extract_from_file()` 内部按 `cfg.retries` 循环调用后端，每次失败间隔 50ms；
重试耗尽后返回 `GarmentResult(ok=False, error={...error_code...})` —— **本函数保证不抛异常**，
方便上层统一按 `result.error["error_code"]` 决定是"提示重试"还是"提示换图"。

---

## 4. 标准化数据结构（要求 3）

### 4.1 `Category`（`str, Enum`，开放枚举）

`top` / `bottom` / `dress` / `outerwear` / `accessory` / `unknown`

成员可与字符串直接比较（`Category.TOP == "top"` 为 `True`）；接分类模型时可自由扩充。

### 4.2 `GarmentItem`（单件衣物）—— **下游业务对接契约**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `item_id` | `str` | 本次提取内唯一 ID（`<session>-<序号>`） |
| `category` | `Category` | 衣物类别 |
| `confidence` | `float` | 置信度 0~1 |
| `bbox` / `bbox_norm` | `BBox` / `dict` | 位置坐标：像素 / 归一化（`x,y,w,h`） |
| `polygon` / `polygon_norm` | `list[[x,y]]` | **轮廓信息**：像素 / 归一化多边形 |
| `mask_png` | `str` | 该件的二值掩码 PNG（裁剪到 bbox，白=衣物） |
| `rgba_png` | `str` | 该件的抠图 PNG（裁剪到 bbox，**透明背景 BGRA**） |
| `attributes` | `dict` | 补充属性：`aspect_w_over_h` / `area_px` / `fill_ratio` / `precision` |
| `source` | `str` | 产出它的后端名 |

`bbox_norm` / `polygon_norm` 用**原图宽高**归一化，所以坐标与图片分辨率无关，
可直接喂给任意尺寸的编辑画布 / 搭配预览。`to_dict()` 输出示例：

```json
{
  "item_id": "20261017-162328-be1858-00",
  "category": "dress",
  "confidence": 0.587,
  "bbox": {"x": 420, "y": 1244, "w": 188, "h": 494},
  "bbox_norm": {"x": 0.328125, "y": 0.686913, "w": 0.146875, "h": 0.272777},
  "polygon": [[495, 1244], [462, 1258], [420, 1339]],
  "polygon_norm": [[0.3867, 0.6869], [0.3609, 0.6946], [0.3281, 0.7393]],
  "mask_png": ".../items/item_00_mask.png",
  "rgba_png": ".../items/item_00_rgba.png",
  "attributes": {"aspect_w_over_h": 0.381, "area_px": 65121, "fill_ratio": 0.7, "precision": "balanced"},
  "source": "white_bg"
}
```

### 4.3 `GarmentResult`（一次提取的整体结果）

`ok` / `source_image` / `width` / `height` / `items` / `backend` / `elapsed_ms` /
`error`（失败时是 `UploadError.to_dict()`）/ `session` / `out_dir` / `warnings`。
`to_dict()` 额外带 `schema_version: "garment.v1"` 与 `item_count`。

### 4.4 `UploadedImage`（一次成功上传）

`path`（临时目录里的副本）/ `original_name` / `size_bytes` / `sha1` / `w` / `h` / `ext` / `session`。

---

## 5. 提取器接口契约 + AI 模型接入指南（要求 2）

### 5.1 后端接口

```python
class BaseExtractor(ABC):
    name = "base"
    @classmethod
    def available(cls) -> bool: ...            # 当前环境是否可用（缺依赖/模型时 False）
    @abstractmethod
    def extract(self, bgr, cfg: UploadConfig, out_dir: str) -> list[GarmentItem]: ...
```

约定：

* 必须返回 `list[GarmentItem]`，且每件的 `mask_png` / `rgba_png` **已经写到 `out_dir`**；
* 坐标一律用**原图像素坐标**（若你内部降采样了，记得按比例映射回去）；
* 抛 `ExtractionError` 表示"这轮失败"，上层会按 `retries` 重试；
* 注册方式：类上加 `@register_extractor`，或 `register_extractor(MyExtractor)`。

```python
@register_extractor
class MyExtractor(BaseExtractor):
    name = "my_backend"
    @classmethod
    def available(cls): return os.path.exists("weights.pt")
    def extract(self, bgr, cfg, out_dir): ...
```

内置后端：`white_bg`（默认可用的白底抠图）、`ai_stub`（AI 模型对接桩，当前 `available()=False`）。

### 5.2 接 AI 模型：只改 3 个钩子

`AiExtractorStub` 已经把一次提取拆成标准三段式，接模型时**只需实现这三个方法**：

| 钩子 | 签名 | 建议接的模型 |
| --- | --- | --- |
| `_detect` | `(bgr, cfg) -> [{"bbox":[x,y,w,h], "score":float, "label":str}, ...]` | YOLO / RT-DETR / Detectron2；或 OWL-ViT 做开放词汇 |
| `_segment` | `(bgr, box, cfg) -> (mask_uint8_hw, polygon)` | **SAM / SAM2** / U2Net / 通用抠图模型 |
| `_classify` | `(bgr, box, mask, cfg) -> (Category, confidence)` | 服装分类器 / CLIP 零样本分类 |

另外还要补 `available()`（判断权重与依赖是否就绪）与 `extract()` 尾部的"组装"步骤
（把 `(框, 掩码, 轮廓, 类别)` 组装成 `GarmentItem` 并导出 mask/rgba PNG，可参考
`WhiteBgExtractor.extract()` 里 `_build_rgba` / `_write_unicode` 那段）。

接好后把 `UploadConfig.extractor` 指向它即可，**主程序与下游业务代码一行都不用改**。
当前直接选 `ai_stub` 会拿到明确的 `E_BACKEND_UNAVAILABLE` 提示（而不是崩溃），提示里就写着该去实现哪几个方法。

### 5.3 默认后端 `white_bg` 的算法（了解它与 AI 后端的差异）

1. 按 `precision` 把原图缩到工作分辨率（省时，坐标最后映射回原图）；
2. **白底抠图**：四边补一圈白 + 一次 `floodFill`（`FIXED_RANGE`）漫水取背景，
   再用"灰度 < `white_thr` 强制实心"兜底 —— 不能改用全局白阈值，否则浅色衣物会被挖出洞；
3. 连通域拆分 → 按 `min_area` 过滤 / 按面积排序 / 限量 `max_items`；
4. 每件：`boundingRect` 出 bbox、`findContours`+`approxPolyDP` 出轮廓多边形、
   按长宽比做**启发式**类别判定、导出 mask/rgba PNG；
5. 坐标全部映射回原图后组装 `GarmentItem`。

---

## 6. 临时存储约定（要求 4）

```
<tmp_dir>/tmp/<session>/                 # session 名 = YYYYMMDD-HHMMSS-<6位随机>
├── <safe-name>-<uuid>.png               # 上传原图副本（保留原格式，不二次编码）
├── upload.json                          # 上传 meta：原名/大小/sha1/尺寸/时间
├── result.json                          # 提取结果（GarmentResult.to_dict()，即 garment.v1）
└── items/
    ├── item_00_mask.png                 # 单件掩码（裁剪到 bbox）
    ├── item_00_rgba.png                 # 单件抠图（透明背景）
    └── ...
```

* `TempStore.new_session()` 建 session；`item_dir()` 建 items 子目录；
* `TempStore.cleanup(ttl_hours=None)` 删除 mtime 超过 ttl 的旧 session，返回被删列表；
  CLI 里是 `--cleanup`，默认按 `session_ttl_hours=24` 回收。
* 临时目录是**中间产物**，不是交付物；正式素材仍只放 `assets/clothes/`。

---

## 7. 与主程序的对接（要求 3 的"无缝对接"证明）

主程序 [main.py](file:///c:/Users/HP/Desktop/TRAE_work/main.py) 只做了**轻量挂载**，原有行为完全不变：

| 入口 | 作用 |
| --- | --- |
| `python main.py --upload 样图.jpg` | 上传 + 提取，打印标准化 JSON 后退出（退出码 0/1） |
| `python main.py --upload 样图.jpg --upload-add` | 提取后把衣物**注入衣柜内存**，再进入摄像头模式 |
| `--upload-preview` / `--no-upload-preview` | 上传前预览开关（默认开） |
| `--upload-backend NAME` | 指定提取后端 |
| 摄像头窗口按 `u` | 弹文件对话框上传 → 提取 → 追加进衣柜，随后按 `n/p` 就能穿上 |

对接点在 `Wardrobe.add_slot_from_bgra(bgra, name, params, rel)`：
它把一件抠图（BGRA 数组）按**与静态资源完全相同的校验和缓存字段**
（`src_a` / `src_rgb` / `thumb`）装进 `wardrobe.slots`，于是 `overlay_item()`
可以像对待普通衣柜条目一样直接渲染它 —— 这就是"提取结果无缝接入现有叠加业务"。

实测（`assets/refs/ref_looks_collage.jpg`）：上传后检出 **8** 件，`_add_extracted_to_wardrobe()`
追加 **8** 件，衣柜 15 → 23 件，8 件全部可渲染（每件在 960x540 帧上绘制 6.8~9.5 万像素，无渲染失败）。

---

## 8. 独立测试方法（要求 6）

```powershell
# ① 模块自检：合成样图跑完整流水线 + 4 类失败用例 + 临时存储/清理，退出码 0/1
.\\.venv\\Scripts\\python.exe garment_upload.py --self-test
```

自检覆盖的断言：

1. 正常上传：`UploadedImage` 字段齐全、原图已复制进 session；
2. 提取流水线：检出数量 = 2、`GarmentItem` 11 个字段齐全、`bbox` 子字段齐全、
   `mask_png`/`rgba_png` 真实落盘、抠图可读回成 4 通道 BGRA（下游对接可用）；
3. 异常：`.txt` 格式非法 → `E_FORMAT`；损坏 PNG → `E_DECODE`；2MB>0.5MB 上限 → `E_SIZE`；
   10x10 → `E_DIMENSION`；不存在 → `E_NOT_FOUND`；类型范围过滤 → `E_NO_GARMENT`；
   未注册后端 → `E_BACKEND`；AI 桩未接 → `E_BACKEND_UNAVAILABLE`；
4. 临时存储：session 内有 `upload.json` 与 `items/*`；`cleanup(ttl=0)` 能回收。

证据图：`upload_selftest_sheet.jpg`（原图上的 bbox + 轮廓 + 类别标签，右侧是每件的透明抠图预览）。

**回归验证原功能未受影响**：

```powershell
.\\.venv\\Scripts\\python.exe main.py --selftest      # 15 件 PASS，退出码 0
.\\.venv\\Scripts\\python.exe main.py --upload assets\\refs\\ref_looks_collage.jpg --no-upload-preview
```

---

## 9. 扩展点速查

| 想做的事 | 改哪里 |
| --- | --- |
| 调识别精度 | `UploadConfig.precision` 或 `extractor_params` 里的 `tol/white_thr/eps/min_area/long_side` |
| 限定衣物类型范围 | `UploadConfig.allowed_categories = ("top","outerwear")` |
| 换/加一个提取后端 | 写 `BaseExtractor` 子类 + `@register_extractor`，`UploadConfig.extractor` 指过去 |
| 接检测/分割/分类 AI 模型 | 实现 `AiExtractorStub._detect / _segment / _classify` |
| 加新的上传校验规则 | `validate_image()` 里追加一条 → 新增一个错误码 |
| 数据库字段不够用 | `GarmentItem.attributes` 里加键；或给 dataclass 加字段并同步 `to_dict()` |
| 换掉系统文件对话框 | 替换 `pick_file_dialog()` 的返回值来源即可（接口只要求"返回一个路径或 None"） |
| 关掉整个上传功能 | `UploadConfig(enabled=False)` |

---

## 10. 已知限制

* `white_bg` 后端只适合**纯色/白底平铺样图**。真实场景照（复杂背景、人台、多件重叠）需要接 AI 后端。
* 浅色衣物（灰度 ≳ `white_thr`，即接近底色）可能被漫水误当背景而漏检 —— 这是该算法的固有弱点，
  也是 `AiExtractorStub` 存在的原因。
* 类别判定是**长宽比启发式**（`h/w≥1.9→dress`、`≥0.85→top`、`≥0.4→bottom`、否则 `accessory`），
  对整套穿搭拼图这类"上衣+裤装连成一块"的图会整块判成 `dress`，属预期行为，真分类请接分类模型。
* 相邻/接触的衣物会被连通域合并成一件（把两件分开需要分割模型或人工框选）。
* 预览窗口是 `cv2.imshow` 阻塞式交互；无 GUI 环境（CI / 无头）会自动跳过预览并视为已确认。
