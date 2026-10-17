# -*- coding: utf-8 -*-
"""
用户上传样图 —— 图片上传 + 衣物元素自动提取（可独立运行的预留功能模块）
=========================================================================
模块定位
--------
本模块为"用户上传样图"这条产品链路提供**完整且可独立测试**的骨架，共 6 部分：

  1. 上传入口：`pick_file_dialog()`（系统文件选择框）+ `preview_image()`（上传前预览，
     Enter/Space 确认、ESC 取消）+ `validate_image()`（格式 / 大小 / 尺寸 / 可解码校验）。
  2. 提取模块：`BaseExtractor` 标准接口 + 提取器注册表。默认后端 `WhiteBgExtractor`
     （白底/纯色底抠图，纯 CPU、无需模型）；`AiExtractorStub` 是**给 AI 模型预留的对接桩**，
     里面按"检测 -> 分割 -> 分类"三段式留了钩子与 TODO，接模型时只改这三个方法。
  3. 标准化数据契约：`GarmentItem`（类别 / 位置坐标 bbox / 轮廓 polygon / 掩码 / 抠图 PNG /
     属性）与 `GarmentResult`，都提供 `to_dict()`，可直接 JSON 化交给下游
     "衣物编辑 / 替换 / 搭配"业务。
  4. 临时存储：`TempStore` 按 session 目录落盘"上传原图 + meta.json + 每件衣物的
     mask.png / rgba.png"，并提供过期 session 清理。
  5. 异常与重试：`UploadError`(基类) -> `ValidationError` / `ExtractionError`，都带
     `error_code`；`extract_from_file()` 按 `UploadConfig.retries` 自动重试。
  6. 功能开关与参数配置：`UploadConfig`（`enabled` 总开关、`precision` 识别精度、
     `allowed_categories` 支持类型范围、`extractor` 后端选择、`extractor_params` 透传）。

为什么是"桌面等价物"
--------------------
本项目是 **Python + OpenCV 桌面原型，没有 Web UI**。因此 Web 语境的
"上传控件 / 拖拽 / 浏览器预览"在这里落成**等价且可验证**的桌面实现：
系统文件选择对话框 = 上传入口，OpenCV 窗口 = 上传前预览，异常提示走控制台文本。

独立测试
--------
    python garment_upload.py --self-test          # 合成样图跑完整流水线 + 3 类失败用例
    python garment_upload.py --image 样图.jpg      # 对真实图片提取并打印标准化 JSON
    python garment_upload.py --list-backends       # 看有哪些提取后端
    python garment_upload.py --cleanup             # 清理过期的临时 session

依赖：仅标准库 + opencv(cv2) + numpy。不 import main.py，可单独跑。

接口速查见 docs/UPLOAD_MODULE.md。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TMP_DIR = os.path.join(BASE_DIR, "assets", "uploads")

__all__ = [
    "UploadConfig", "Category", "BBox", "GarmentItem", "GarmentResult", "UploadedImage",
    "UploadError", "ValidationError", "ExtractionError",
    "BaseExtractor", "WhiteBgExtractor", "AiExtractorStub",
    "register_extractor", "get_extractor", "list_extractors",
    "TempStore", "validate_image", "pick_file_dialog", "preview_image",
    "upload_image", "extract_from_file", "upload_and_extract", "load_item_bgra",
]


# ============================================================================
# 0. 配置（要求 5：功能开关 + 参数配置项）
# ============================================================================
@dataclass
class UploadConfig:
    """上传/提取模块的全部开关与参数，集中在这里调，不改代码逻辑。

    enabled             : 总功能开关。False 时所有上传入口直接报 E_DISABLED。
    allowed_formats     : 允许的图片扩展名（小写、不含点）。
    max_file_mb         : 单文件大小上限（MB）。
    min_side_px         : 图片最小边下限（像素）。
    max_side_px         : 图片最大边上限（像素，防超大图打爆内存）。
    tmp_dir             : 上传/提取产物的临时根目录（session 子目录都建在它下面）。
    session_ttl_hours   : 临时 session 的保留时长，超过即被 cleanup() 回收。
    preview             : 是否启用"上传前预览确认"。
    retries             : 提取失败后的自动重试次数（0 = 不重试）。
    precision           : 识别精度档位 fast / balanced / high。
    allowed_categories  : 允许保留的衣物类型范围（扩展点：只留上衣，或放开全部）。
    extractor           : 提取后端名，见 list_extractors()。
    extractor_params    : 透传给后端 extract() 的额外参数（覆盖 precision 档位里的同名键）。
    """

    enabled: bool = True
    allowed_formats: tuple = ("jpg", "jpeg", "png", "webp", "bmp")
    max_file_mb: float = 12.0
    min_side_px: int = 64
    max_side_px: int = 8000
    tmp_dir: str = DEFAULT_TMP_DIR
    session_ttl_hours: float = 24.0
    preview: bool = True
    retries: int = 1
    precision: str = "balanced"
    # 类型范围扩展点：默认放开全部；例如只想要上衣可设 ("top", "outerwear")
    allowed_categories: tuple = ("top", "bottom", "dress", "outerwear", "accessory", "unknown")
    extractor: str = "white_bg"
    extractor_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["allowed_formats"] = list(self.allowed_formats)
        d["allowed_categories"] = list(self.allowed_categories)
        return d


# 识别精度档位：影响处理分辨率、抠图容差、轮廓简化强度、最小面积过滤
PRECISION_PRESETS = {
    #            处理长边  抠图容差  灰度"强制实心"阈值  轮廓简化 eps  最小面积占比  填充孔洞
    "fast":     {"long_side": 640,  "tol": 22, "white_thr": 205, "eps": 0.020, "min_area": 0.020, "close_it": 2},
    "balanced": {"long_side": 1280, "tol": 14, "white_thr": 205, "eps": 0.010, "min_area": 0.012, "close_it": 2},
    "high":     {"long_side": 2200, "tol": 9,  "white_thr": 208, "eps": 0.005, "min_area": 0.008, "close_it": 3},
}


# ============================================================================
# 1. 异常体系（要求 4：异常处理）
# ============================================================================
class UploadError(Exception):
    """本模块所有可预期错误的基类；带 `code` 便于上层做分支提示与重试判断。"""

    code = "E_UPLOAD"

    def __init__(self, message: str, code: str | None = None, detail=None):
        super().__init__(message)
        self.message = message
        self.code = code or type(self).code
        self.detail = detail

    def to_dict(self) -> dict:
        return {"ok": False, "error_code": self.code, "error": self.message, "detail": self.detail}


class ValidationError(UploadError):
    """上传阶段的校验失败：格式 / 大小 / 尺寸 / 损坏 / 用户取消。"""

    code = "E_VALIDATE"


class ExtractionError(UploadError):
    """提取阶段的失败：后端不可用 / 模型未接 / 未识别到衣物 / 后端内部异常。"""

    code = "E_EXTRACT"


# 错误码速查（详见 docs/UPLOAD_MODULE.md）
ERROR_CODES = {
    "E_DISABLED": "上传功能被 UploadConfig.enabled=False 关闭",
    "E_CANCELLED": "用户取消（未选文件，或预览时按 ESC）",
    "E_NOT_FOUND": "文件不存在",
    "E_NOT_FILE": "给定路径不是普通文件",
    "E_FORMAT": "扩展名不在 allowed_formats 白名单内",
    "E_EMPTY": "文件长度为 0",
    "E_SIZE": "超过 max_file_mb 大小上限",
    "E_DECODE": "解码失败：不是有效图片或已损坏",
    "E_DIMENSION": "尺寸超出 min_side_px / max_side_px 限制",
    "E_TMP_WRITE": "写入临时目录失败",
    "E_BACKEND": "未注册的提取后端名",
    "E_BACKEND_UNAVAILABLE": "后端在当前环境不可用（如缺模型/缺依赖）",
    "E_NO_GARMENT": "未在图片中识别到衣物元素",
    "E_EXTRACT": "提取过程发生内部异常（重试耗尽后抛出）",
}


# ============================================================================
# 2. 标准化数据契约（要求 3）
# ============================================================================
class Category(str, Enum):
    """衣物类别。这是**开放枚举**，接 AI 分类模型时可自由扩充成员。"""

    TOP = "top"              # 上装
    BOTTOM = "bottom"        # 下装
    DRESS = "dress"          # 连衣裙 / 连体装
    OUTERWEAR = "outerwear"  # 外套
    ACCESSORY = "accessory"  # 配饰（包 / 帽 / 鞋 ...）
    UNKNOWN = "unknown"      # 无法判定


@dataclass
class BBox:
    """轴对齐包围盒（整图像素坐标）。"""

    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict:
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h)}

    def to_norm(self, img_w: int, img_h: int) -> dict:
        iw, ih = max(1, img_w), max(1, img_h)
        return {"x": round(self.x / iw, 6), "y": round(self.y / ih, 6),
                "w": round(self.w / iw, 6), "h": round(self.h / ih, 6)}


@dataclass
class GarmentItem:
    """单件衣物的标准化结果。字段即"与下游业务对接"的契约。

    item_id       : 本次提取内的唯一 ID（session 前缀 + 序号）
    category      : Category 枚举
    confidence    : 置信度 0~1
    bbox / bbox_norm      : 像素包围盒 / 归一化包围盒
    polygon / polygon_norm: 外轮廓多边形（像素 / 归一化）
    mask_png      : 该件衣物的二值掩码 PNG（裁剪到 bbox，白=衣物）
    rgba_png      : 该件衣物的抠图 PNG（裁剪到 bbox，透明背景）
    attributes    : 后端的补充属性（长宽比 / 面积 / 填充率 等，供后续编辑、搭配使用）
    source        : 产出它的后端名
    """

    item_id: str
    category: Category
    confidence: float
    bbox: BBox
    bbox_norm: dict
    polygon: list
    polygon_norm: list
    mask_png: str
    rgba_png: str
    attributes: dict
    source: str

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "category": self.category.value,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "bbox_norm": dict(self.bbox_norm),
            "polygon": [[int(p[0]), int(p[1])] for p in self.polygon],
            "polygon_norm": [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in self.polygon_norm],
            "mask_png": self.mask_png,
            "rgba_png": self.rgba_png,
            "attributes": dict(self.attributes),
            "source": self.source,
        }


@dataclass
class GarmentResult:
    """一次"从图片提取衣物"的完整结果，成功/失败都返回它，便于统一处理与重试。"""

    ok: bool
    source_image: str
    width: int
    height: int
    items: list                       # list[GarmentItem]
    backend: str
    elapsed_ms: float
    error: dict | None = None
    session: str | None = None
    out_dir: str | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": "garment.v1",
            "ok": bool(self.ok),
            "source_image": self.source_image,
            "width": int(self.width),
            "height": int(self.height),
            "backend": self.backend,
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "item_count": len(self.items),
            "items": [it.to_dict() for it in self.items],
            "session": self.session,
            "out_dir": self.out_dir,
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass
class UploadedImage:
    """一次成功上传的产物信息（原图已复制进 session 临时目录）。"""

    path: str
    original_name: str
    size_bytes: int
    sha1: str
    w: int
    h: int
    ext: str
    session: str

    def to_dict(self) -> dict:
        return {
            "path": self.path, "original_name": self.original_name,
            "size_bytes": self.size_bytes, "sha1": self.sha1,
            "width": self.w, "height": self.h, "ext": self.ext, "session": self.session,
        }


# ============================================================================
# 3. 小工具：中文路径安全的读写
# ============================================================================
def _imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED):
    """cv2.imread 不支持中文路径，用 imdecode 兜底。"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception:
        return None


def _imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    """cv2.imwrite 不支持中文路径，用 imencode + tofile 兜底。"""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sha1_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_.\-]+", "_", os.path.splitext(os.path.basename(name))[0])
    return (s[:40] or "image").strip("_") or "image"


def _fit_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """等比缩放到不超过 (max_w, max_h)，用于预览窗口。"""
    h, w = img.shape[:2]
    s = min(1.0, max_w / float(w), max_h / float(h))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


# ============================================================================
# 4. 上传入口（要求 1）：文件对话框 / 预览 / 校验
# ============================================================================
def pick_file_dialog(title: str | None = None, initialdir: str | None = None) -> str | None:
    """弹出系统文件选择框选一张图片；环境无 GUI / 用户取消时返回 None（优雅降级）。

    这是"上传交互入口"的桌面等价物。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print("[上传] 当前 Python 没有可用的 tkinter，无法弹出文件选择框。")
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(
            title=title or "选择要上传的样图",
            initialdir=initialdir or BASE_DIR,
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"),
                ("WEBP", "*.webp"), ("BMP", "*.bmp"), ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        return path or None
    except Exception as e:
        print(f"[上传] 文件选择框不可用（{e}），已降级为取消。")
        return None


def preview_image(path: str, cfg: "UploadConfig | None" = None,
                  window_name: str = "上传预览  -  Enter/Space 确认, ESC 取消") -> bool:
    """上传前预览：等比显示图片，Enter/Space 确认，ESC 取消。

    返回 True = 确认上传；False = 用户取消。无 GUI 环境时打印提示并视为"确认"，
    保证非交互场景（CI / 自检）也能跑通。
    """
    cfg = cfg or UploadConfig()
    if not cfg.preview:
        return True
    img = _imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValidationError("预览失败：图片无法解码", "E_DECODE")
    h, w = img.shape[:2]
    show = _fit_to_screen(img, 1000, 700)
    banner = f"{os.path.basename(path)}   {w}x{h}   Enter/Space 确认  ESC 取消"
    try:
        canvas = show.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(canvas, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (13, 10, 32):        # Enter / Space
                return True
            if k == 27:                  # ESC
                return False
    except cv2.error:
        print("[预览] 当前环境没有 GUI 窗口，跳过预览（视为已确认）。")
        return True
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


def validate_image(path: str | None, cfg: "UploadConfig | None" = None) -> dict:
    """上传前的完整校验（存在性 / 扩展名 / 大小 / 尺寸 / 可解码），失败抛 ValidationError。

    成功返回 {path, ext, size_bytes, w, h, sha1}。
    """
    cfg = cfg or UploadConfig()
    if not cfg.enabled:
        raise ValidationError("上传功能已在配置中关闭（UploadConfig.enabled=False）", "E_DISABLED")
    if not path:
        raise ValidationError("未选择文件", "E_CANCELLED")
    if not os.path.exists(path):
        raise ValidationError(f"文件不存在: {path}", "E_NOT_FOUND")
    if not os.path.isfile(path):
        raise ValidationError(f"给定路径不是普通文件: {path}", "E_NOT_FILE")

    ext = os.path.splitext(path)[1].lstrip(".").lower()
    allowed = tuple(x.lower() for x in cfg.allowed_formats)
    if ext not in allowed:
        raise ValidationError(
            f"不支持的格式 .{ext or '(无扩展名)'}，允许: {', '.join(allowed)}", "E_FORMAT")

    size = os.path.getsize(path)
    if size <= 0:
        raise ValidationError("文件为空（0 字节）", "E_EMPTY")
    limit = int(round(cfg.max_file_mb * 1024 * 1024))
    if size > limit:
        raise ValidationError(
            f"文件过大 {size / 1048576:.2f}MB > 上限 {cfg.max_file_mb}MB", "E_SIZE")

    img = _imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValidationError("解码失败：不是有效图片格式或文件已损坏", "E_DECODE")
    h, w = img.shape[:2]
    if min(h, w) < int(cfg.min_side_px):
        raise ValidationError(f"图片过小 {w}x{h}：最小边需 >= {cfg.min_side_px}px", "E_DIMENSION")
    if max(h, w) > int(cfg.max_side_px):
        raise ValidationError(f"图片过大 {w}x{h}：最大边需 <= {cfg.max_side_px}px", "E_DIMENSION")

    return {"path": os.path.abspath(path), "ext": ext, "size_bytes": size,
            "w": w, "h": h, "sha1": _sha1_file(path)}


# ============================================================================
# 5. 临时存储（要求 4）
# ============================================================================
class TempStore:
    """上传/提取产物的临时存储：按 session 目录组织，并提供过期清理。

    目录约定：
        <tmp_dir>/tmp/<session>/
            ├── <safe-name>-<uuid>.png      # 上传原图的副本（保留原格式，不二次编码）
            ├── upload.json                 # 上传 meta（原名/大小/sha1/尺寸）
            ├── result.json                 # 提取结果（GarmentResult.to_dict()）
            └── items/
                ├── item_00_mask.png        # 单件掩码（裁剪到 bbox）
                ├── item_00_rgba.png        # 单件抠图（透明背景）
                └── ...
    """

    def __init__(self, root: str | None = None, ttl_hours: float = 24.0):
        self.root = os.path.abspath(root or DEFAULT_TMP_DIR)
        self.ttl_hours = float(ttl_hours)
        self._session: str | None = None

    def new_session(self) -> str:
        """创建一个新 session 目录并返回其绝对路径。"""
        name = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        d = os.path.join(self.root, "tmp", name)
        try:
            os.makedirs(os.path.join(d, "items"), exist_ok=True)
        except OSError as e:
            raise UploadError(f"无法创建临时目录 {d}: {e}", "E_TMP_WRITE")
        self._session = d
        return d

    @property
    def session(self) -> str | None:
        return self._session

    def item_dir(self, session: str | None = None) -> str:
        d = os.path.join(session or self._session or self.new_session(), "items")
        os.makedirs(d, exist_ok=True)
        return d

    def save_image(self, session: str, src_path: str, original_name: str | None = None) -> str:
        """把上传的原图**复制**进 session 目录（保留原格式，避免二次编码损失）。"""
        ext = (os.path.splitext(src_path)[1] or ".png").lower()
        stem = _safe_stem(original_name or src_path)
        dst = os.path.join(session, f"{stem}-{uuid.uuid4().hex[:8]}{ext}")
        try:
            shutil.copy2(src_path, dst)
        except OSError as e:
            raise UploadError(f"写入临时目录失败: {e}", "E_TMP_WRITE")
        return dst

    def write_meta(self, session: str, name: str, obj) -> str:
        p = os.path.join(session, name)
        try:
            _write_json(p, obj)
        except OSError as e:
            raise UploadError(f"写入 meta 失败: {e}", "E_TMP_WRITE")
        return p

    def sessions(self) -> list:
        base = os.path.join(self.root, "tmp")
        if not os.path.isdir(base):
            return []
        return sorted(os.path.join(base, n) for n in os.listdir(base)
                      if os.path.isdir(os.path.join(base, n)))

    def cleanup(self, ttl_hours: float | None = None) -> list:
        """删除 mtime 早于 ttl 的旧 session，返回被删目录列表。"""
        ttl = float(self.ttl_hours if ttl_hours is None else ttl_hours)
        now = time.time()
        removed = []
        for d in self.sessions():
            try:
                if (now - os.path.getmtime(d)) > ttl * 3600.0:
                    shutil.rmtree(d, ignore_errors=True)
                    removed.append(d)
            except OSError:
                pass
        return removed


# ============================================================================
# 6. 提取器接口（要求 2：标准化算法对接接口）
# ============================================================================
class BaseExtractor(ABC):
    """所有衣物提取后端的统一接口。接新模型 = 新写一个子类并 register_extractor。

    约定：
      * `extract()` 必须返回 list[GarmentItem]，且每个 item 的 mask/rgba PNG 必须
        已经写到 out_dir 里（路径写进 item.mask_png / item.rgba_png）。
      * 坐标系一律用**原图像素坐标**（未知原图尺寸时可自行 scale 回原尺寸）。
      * 抛 ExtractionError（或子类）表示"这轮提取失败"，上层会按 retries 重试。
    """

    name = "base"

    @classmethod
    def available(cls) -> bool:
        """后端在当前环境是否可用（缺依赖/缺模型时返回 False）。"""
        return True

    @abstractmethod
    def extract(self, bgr: np.ndarray, cfg: UploadConfig, out_dir: str) -> list:
        """从 BGR 原图中提取衣物元素。"""
        raise NotImplementedError


_EXTRACTORS: dict = {}


def register_extractor(cls: type) -> type:
    """把提取器类登记到全局注册表（可作装饰器用）。"""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError("提取器必须有唯一的 name 属性")
    _EXTRACTORS[name] = cls
    return cls


def get_extractor(name: str) -> BaseExtractor:
    cls = _EXTRACTORS.get(name)
    if cls is None:
        raise ExtractionError(
            f"未注册的提取后端 '{name}'，可用: {', '.join(sorted(_EXTRACTORS))}", "E_BACKEND")
    return cls()


def list_extractors() -> list:
    return [{"name": n, "available": bool(c.available())}
            for n, c in sorted(_EXTRACTORS.items())]


# --------------------------- 6.1 默认后端：白底/纯色底抠图 ---------------------------
@register_extractor
class WhiteBgExtractor(BaseExtractor):
    """默认后端：适合"白底平铺样图"（无模型、纯 CPU、开箱即用）。

    流程：
      1. 按 precision 档位把原图缩到工作分辨率（省时，坐标最后再映射回原图）；
      2. 白底抠图 `_foreground_mask()`：**四边补白 + 一次 floodFill 漫水** 得到背景，
         再用"灰度 < white_thr 强制实心"兜底（防止浅色衣物被挖出洞）；
      3. 连通域拆分 -> 按面积过滤/排序/限量；
      4. 每件：求 bbox、`findContours` + `approxPolyDP` 得轮廓多边形、按长宽比做
         **启发式类别判定**、导出 mask.png 与 rgba.png（透明背景）；
      5. 坐标全部映射回原图尺寸后组装 GarmentItem。
    """

    name = "white_bg"

    @classmethod
    def available(cls) -> bool:
        return True

    def extract(self, bgr: np.ndarray, cfg: UploadConfig, out_dir: str) -> list:
        if bgr is None or bgr.ndim != 3:
            raise ExtractionError("输入不是有效的 BGR 图像", "E_EXTRACT")
        params = dict(PRECISION_PRESETS.get(cfg.precision, PRECISION_PRESETS["balanced"]))
        params.update(cfg.extractor_params or {})   # 扩展点：逐项覆盖
        # 防止把精度档位里不认识的键透传下去（只保留已知键，其余当扩展参数忽略）
        long_side = int(params.get("long_side", 1280))
        tol = float(params.get("tol", 14))
        white_thr = float(params.get("white_thr", 205))
        eps = float(params.get("eps", 0.01))
        min_area_ratio = float(params.get("min_area", 0.012))
        close_it = int(params.get("close_it", 2))
        max_items = int(params.get("max_items", 8))

        H0, W0 = bgr.shape[:2]
        work, scale = self._to_work_size(bgr, long_side)
        fg = self._foreground_mask(work, tol, white_thr, close_it)
        h, w = fg.shape[:2]

        num, labels, stats, _cents = cv2.connectedComponentsWithStats(fg, 8)
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        total = float(sum(areas)) or 1.0
        min_px = max(200, int(min_area_ratio * h * w))
        keep = [i for i in range(1, num)
                if stats[i, cv2.CC_STAT_AREA] >= min_px
                and stats[i, cv2.CC_STAT_AREA] >= min_area_ratio * total]
        keep.sort(key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
        keep = keep[:max_items]
        if not keep:
            raise ExtractionError("未在该图片中识别到衣物元素（背景非纯色 / 衣物与背景太接近）",
                                  "E_NO_GARMENT")

        items, warnings = [], []
        allowed = {c.lower() for c in cfg.allowed_categories}
        for rank, i in enumerate(keep):
            x, y, bw, bh, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
                                  int(stats[i, cv2.CC_STAT_AREA]))
            if bw < 4 or bh < 4:
                continue
            comp = np.where(labels[y:y + bh, x:x + bw] == i, 255, 0).astype(np.uint8)
            category = self._classify(bw, bh)
            if category.value not in allowed:
                warnings.append(f"跳过类别 {category.value}（不在 allowed_categories 内）")
                continue

            # 工作分辨率 -> 原图坐标
            xo, yo = int(round(x / scale)), int(round(y / scale))
            bwc = min(W0 - xo, max(1, int(round(bw / scale))))
            bhc = min(H0 - yo, max(1, int(round(bh / scale))))
            bbox = BBox(xo, yo, bwc, bhc)

            poly_w = self._polygon(comp, eps)
            poly = [[int(round(px / scale)) + xo, int(round(py / scale)) + yo] for px, py in poly_w]

            src_crop = work[y:y + bh, x:x + bw]
            rgba = self._build_rgba(src_crop, comp)
            mask_path = os.path.join(out_dir, f"item_{rank:02d}_mask.png")
            rgba_path = os.path.join(out_dir, f"item_{rank:02d}_rgba.png")
            if not _imwrite_unicode(mask_path, comp):
                raise ExtractionError(f"写入掩码失败: {mask_path}", "E_TMP_WRITE")
            if not _imwrite_unicode(rgba_path, rgba):
                raise ExtractionError(f"写入抠图失败: {rgba_path}", "E_TMP_WRITE")

            sess_tag = os.path.basename(os.path.dirname(out_dir.rstrip(os.sep))) or "s"
            conf = round(min(0.99, 0.55 + 0.4 * (area / total)), 3)
            items.append(GarmentItem(
                item_id=f"{sess_tag}-{rank:02d}",
                category=category,
                confidence=conf,
                bbox=bbox,
                bbox_norm=bbox.to_norm(W0, H0),
                polygon=poly,
                polygon_norm=[[round(px / max(1, W0), 6), round(py / max(1, H0), 6)] for px, py in poly],
                mask_png=os.path.abspath(mask_path),
                rgba_png=os.path.abspath(rgba_path),
                attributes={
                    "aspect_w_over_h": round(bw / float(max(1, bh)), 3),
                    "area_px": area,
                    "fill_ratio": round(area / float(max(1, bw * bh)), 3),
                    "precision": cfg.precision,
                },
                source=self.name,
            ))
        if not items:
            raise ExtractionError("检出的候选区域都被过滤掉了（面积过小或类别不在范围内）",
                                  "E_NO_GARMENT")
        self._warnings = warnings
        return items

    # ---------------- 内部：实现细节 ----------------
    @staticmethod
    def _to_work_size(bgr: np.ndarray, long_side: int):
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m <= long_side:
            return bgr, 1.0
        s = long_side / float(m)
        out = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return out, s

    @staticmethod
    def _foreground_mask(bgr: np.ndarray, tol: float, white_thr: float, close_it: int) -> np.ndarray:
        """白底抠图：四边补白 + 一次 floodFill 漫水取背景，再对中低灰度像素强制实心兜底。"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pad = 1
        gp = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        mask = np.zeros((gp.shape[0] + 2, gp.shape[1] + 2), np.uint8)
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(gp, mask, (0, 0), 0, int(round(tol)), int(round(tol)), flags)
        bg = mask[1:-1, 1:-1][pad:-pad, pad:-pad]      # 255 = 从外圈漫到的背景
        fg = ((bg == 0) | (gray < white_thr)).astype(np.uint8) * 255

        k = max(3, (min(fg.shape[:2]) // 200) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker, iterations=max(1, close_it))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker, iterations=1)
        return fg

    @staticmethod
    def _polygon(comp: np.ndarray, eps_ratio: float) -> list:
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, max(1.0, eps_ratio * peri), True)
        return [[float(p[0][0]), float(p[0][1])] for p in ap]

    @staticmethod
    def _build_rgba(bgr_crop: np.ndarray, comp: np.ndarray) -> np.ndarray:
        """把 bbox 裁剪块按掩码做成透明背景 BGRA。"""
        h, w = comp.shape[:2]
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[:, :, :3] = bgr_crop
        rgba[:, :, 3] = comp
        return rgba

    @staticmethod
    def _classify(bw: int, bh: int) -> Category:
        """**启发式**类别判定（只按长宽比粗分）；真分类请交给 AI 后端。

        规则：h/w >= 1.9 -> 连衣裙；>= 0.85 -> 上装；>= 0.4 -> 下装；否则配饰。
        """
        r = bh / float(max(1, bw))
        if r >= 1.9:
            return Category.DRESS
        if r >= 0.85:
            return Category.TOP
        if r >= 0.4:
            return Category.BOTTOM
        return Category.ACCESSORY


# --------------------------- 6.2 AI 模型对接桩（预留） ---------------------------
@register_extractor
class AiExtractorStub(BaseExtractor):
    """**给 AI 模型预留的对接桩**：接口与 WhiteBgExtractor 完全一致，可直接替换。

    它把一次提取拆成三段标准钩子，接真实模型时**只需实现这三个方法**：

        _detect(bgr, cfg)            -> 候选框 list[{"bbox": [x,y,w,h], "score": float}]
                                        推荐接：YOLO / RT-DETR / Detectron2 等**检测**模型
                                        或直接跑 "Fashion-CLIP / OWL-ViT" 做开放词汇检索
        _segment(bgr, box, cfg)      -> (mask_uint8_hw, polygon)  针对单个框
                                        推荐接：SAM / SAM2 / U2Net / 抠图模型
        _classify(bgr, box, mask, cfg)-> (Category, confidence)
                                        推荐接：Fashion-MNIST 系分类器 / CLIP 零样本分类

    当前 `available()` 返回 False（未接模型），`extract()` 会抛
    ExtractionError(E_BACKEND_UNAVAILABLE) 并指明该去实现哪几个方法——
    这样把 `UploadConfig.extractor` 改成 "ai_stub" 时能拿到**明确的错误提示**而不是崩溃。
    """

    name = "ai_stub"

    @classmethod
    def available(cls) -> bool:
        # TODO：接上真实模型后，这里改成"模型文件存在 + 依赖可导入"的真实判断
        return False

    def extract(self, bgr: np.ndarray, cfg: UploadConfig, out_dir: str) -> list:
        if not self.available():
            raise ExtractionError(
                "AI 提取后端尚未接入（AiExtractorStub.available() == False）。\n"
                "接入方式：在 garment_upload.py 的 AiExtractorStub 中实现 "
                "_detect / _segment / _classify 三个钩子，或新写一个 BaseExtractor 子类"
                "并用 register_extractor 注册；再把 UploadConfig.extractor 指向它。\n"
                "详见 docs/UPLOAD_MODULE.md 的『AI 模型接入指南』。",
                "E_BACKEND_UNAVAILABLE")

        H0, W0 = bgr.shape[:2]
        # ---- 1) 检测：拿到候选框 ----
        boxes = self._detect(bgr, cfg)                       # noqa: F841  (TODO 实现)
        # ---- 2) 分割：逐框出掩码 + 轮廓 ----
        # ---- 3) 分类：逐框出类别 + 置信度 ----
        # ---- 4) 组装：把 (框, 掩码, 轮廓, 类别) 组装成 GarmentItem，导出 mask/rgba PNG ----
        raise ExtractionError("AI 后端骨架未完成组装逻辑（见 _detect 之后的三段 TODO）",
                              "E_EXTRACT")

    # ---------------- 预留钩子（接模型时实现这三个） ----------------
    def _detect(self, bgr: np.ndarray, cfg: UploadConfig) -> list:
        """[TODO] 检测钩子：返回候选衣物框 [{"bbox":[x,y,w,h], "score":float, "label":str}, ...]"""
        raise NotImplementedError(
            "接入检测模型：加载权重后对 bgr 推理，把结果转成 [{'bbox':[x,y,w,h],'score':s}, ...]")

    def _segment(self, bgr: np.ndarray, box, cfg: UploadConfig):
        """[TODO] 分割钩子：对单个 box 出像素级掩码，返回 (mask_uint8_hw, polygon)。"""
        raise NotImplementedError(
            "接入分割模型：以 box 为 prompt 跑 SAM/SAM2/U2Net，返回 mask(uint8,0/255) 与轮廓多边形")

    def _classify(self, bgr: np.ndarray, box, mask: np.ndarray, cfg: UploadConfig):
        """[TODO] 分类钩子：返回 (Category, confidence)。"""
        raise NotImplementedError(
            "接入服装分类器（含 CLIP 零样本）：返回 (Category.XXX, 0~1 的置信度)")


# 兼容：把白底后端内部抛出的 warnings 暴露给上层（无状态的一次性字段，仅作提示）
WhiteBgExtractor._warnings = []
AiExtractorStub._warnings = []


# ============================================================================
# 7. 编排：上传 + 提取（要求 2、4 的对外入口）
# ============================================================================
def upload_image(path: str | None = None, cfg: "UploadConfig | None" = None) -> UploadedImage:
    """上传一张样图：校验 -> （可选）预览确认 -> 存入临时 session。

    path=None 时会弹出系统文件选择框。
    失败抛 ValidationError / UploadError（用户取消时为 E_CANCELLED）。
    """
    cfg = cfg or UploadConfig()
    if path is None:
        path = pick_file_dialog()
    info = validate_image(path, cfg)             # 抛 ValidationError（含 E_CANCELLED）

    if cfg.preview and not preview_image(info["path"], cfg):
        raise UploadError("用户取消了上传（预览阶段按 ESC）", "E_CANCELLED")

    store = TempStore(cfg.tmp_dir, cfg.session_ttl_hours)
    session = store.new_session()
    saved = store.save_image(session, info["path"], os.path.basename(info["path"]))
    store.write_meta(session, "upload.json", {
        "original_path": info["path"],
        "original_name": os.path.basename(info["path"]),
        "stored_path": saved,
        "size_bytes": info["size_bytes"],
        "sha1": info["sha1"],
        "width": info["w"],
        "height": info["h"],
        "ext": info["ext"],
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return UploadedImage(path=saved, original_name=os.path.basename(info["path"]),
                         size_bytes=info["size_bytes"], sha1=info["sha1"],
                         w=info["w"], h=info["h"], ext=info["ext"], session=session)


def extract_from_file(path: str, cfg: "UploadConfig | None" = None,
                      backend: str | None = None, out_dir: str | None = None) -> GarmentResult:
    """对一张图片跑衣物提取，**永不抛异常**——成功/失败都返回 GarmentResult。

    失败时 result.ok=False、result.error 是 UploadError.to_dict()（带 error_code），
    内部会按 cfg.retries 重试。
    """
    cfg = cfg or UploadConfig()
    t0 = time.perf_counter()
    name = backend or cfg.extractor

    def _fail(code, msg, detail=None):
        return GarmentResult(ok=False, source_image=os.path.abspath(path) if path else "",
                             width=0, height=0, items=[], backend=name,
                             elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                             error={"ok": False, "error_code": code, "error": msg, "detail": detail},
                             out_dir=out_dir)

    if not cfg.enabled:
        return _fail("E_DISABLED", "上传/提取功能已在配置中关闭（UploadConfig.enabled=False）")

    bgr = _imread_unicode(path, cv2.IMREAD_COLOR) if path else None
    if bgr is None:
        return _fail("E_DECODE", f"读不到图片或解码失败: {path}")

    try:
        extractor = get_extractor(name)
    except UploadError as e:
        return _fail(e.code, e.message)

    session = None
    if out_dir is None:
        store = TempStore(cfg.tmp_dir, cfg.session_ttl_hours)
        session = store.new_session()
        out_dir = store.item_dir(session)
    os.makedirs(out_dir, exist_ok=True)

    H0, W0 = bgr.shape[:2]
    last_err = None
    for attempt in range(int(cfg.retries) + 1):
        try:
            items = extractor.extract(bgr, cfg, out_dir)
        except UploadError as e:
            last_err = e
        except Exception as e:                    # 后端内部未预期异常也不让上层崩
            last_err = ExtractionError(f"{type(e).__name__}: {e}", "E_EXTRACT")
        else:
            result = GarmentResult(
                ok=True, source_image=os.path.abspath(path), width=W0, height=H0,
                items=items, backend=name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                session=session, out_dir=out_dir,
                warnings=list(getattr(extractor, "_warnings", []) or []))
            if session:
                TempStore(cfg.tmp_dir).write_meta(session, "result.json", result.to_dict())
            return result
        if attempt < int(cfg.retries):
            time.sleep(0.05)
    return _fail(last_err.code, f"提取失败（已重试 {cfg.retries} 次）: {last_err.message}")


def upload_and_extract(path: str | None = None, cfg: "UploadConfig | None" = None,
                       backend: str | None = None) -> tuple:
    """一步到位：上传（校验+预览+落临时目录）-> 提取衣物。

    返回 (UploadedImage, GarmentResult)。上传失败会抛 UploadError；
    提取失败不抛，看 GarmentResult.ok / .error。
    """
    cfg = cfg or UploadConfig()
    up = upload_image(path, cfg)
    res = extract_from_file(up.path, cfg, backend=backend,
                            out_dir=os.path.join(up.session, "items"))
    return up, res


def load_item_bgra(item: GarmentItem) -> np.ndarray | None:
    """把 GarmentItem 的抠图 PNG 读成 BGRA 数组（给"编辑 / 替换 / 搭配"下游复用）。"""
    return _imread_unicode(item.rgba_png, cv2.IMREAD_UNCHANGED)


# ============================================================================
# 8. 可视化（自检用）：把 bbox / polygon / 抠图拼成证据图
# ============================================================================
def _checkerboard(h: int, w: int, cell: int = 12) -> np.ndarray:
    ys = (np.arange(h) // cell)[:, None]
    xs = (np.arange(w) // cell)[None, :]
    v = np.where(((ys + xs) % 2) == 0, 220, 160).astype(np.uint8)
    return np.dstack([v, v, v])


def render_visualization(bgr: np.ndarray, result: GarmentResult) -> np.ndarray:
    """在原图上画框+轮廓+类别，右侧拼出每件的透明抠图预览。"""
    canvas = bgr.copy()
    colors = [(0, 255, 0), (255, 200, 0), (0, 160, 255), (255, 0, 200), (0, 255, 255)]
    for i, it in enumerate(result.items):
        col = colors[i % len(colors)]
        b = it.bbox
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.w, b.y + b.h), col, 2, cv2.LINE_AA)
        if len(it.polygon) >= 3:
            cv2.polylines(canvas, [np.array(it.polygon, np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"#{i} {it.category.value} {it.confidence:.2f}",
                    (b.x + 2, max(16, b.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    tiles = []
    for i, it in enumerate(result.items):
        rgba = load_item_bgra(it)
        if rgba is None:
            continue
        h, w = rgba.shape[:2]
        s = min(150.0 / max(1, h), 150.0 / max(1, w), 1.0)
        t = cv2.resize(rgba, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        a = t[:, :, 3:4].astype(np.float32) / 255.0
        bg = _checkerboard(t.shape[0], t.shape[1]).astype(np.float32)
        comp = (t[:, :, :3].astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)
        pad = np.full((comp.shape[0] + 8, comp.shape[1] + 8, 3), 30, np.uint8)
        pad[4:4 + comp.shape[0], 4:4 + comp.shape[1]] = comp
        cv2.putText(pad, f"#{i} {it.category.value}", (6, pad.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(pad)

    if tiles:
        th = max(t.shape[0] for t in tiles)
        tw = sum(t.shape[1] for t in tiles) + 8 * (len(tiles) + 1)
        bar = np.full((th + 16, tw, 3), 30, np.uint8)
        x = 8
        for t in tiles:
            bar[8:8 + t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 8
        if bar.shape[1] != canvas.shape[1]:
            bar = cv2.resize(bar, (canvas.shape[1], max(1, int(bar.shape[0] * canvas.shape[1] / bar.shape[1]))))
        canvas = np.vstack([canvas, bar])
    return canvas


# ============================================================================
# 9. 合成样图（自检用，不依赖任何外部素材）
# ============================================================================
def make_sample_image(w: int = 960, h: int = 640) -> np.ndarray:
    """生成一张"白底 + 两件平铺衣物"的合成样图，用于自检。"""
    img = np.full((h, w, 3), 250, np.uint8)
    # 左：上装（带袖的 T 恤剪影）—— 长宽比约 1.05 -> 判为 top
    cx, sy, hy = int(w * 0.26), int(h * 0.16), int(h * 0.66)
    half = int(w * 0.085)
    body = np.array([[cx - half, sy], [cx + half, sy], [cx + int(half * 0.85), hy],
                     [cx - int(half * 0.85), hy]], np.int32)
    cv2.fillPoly(img, [body], (60, 60, 205), cv2.LINE_8)
    cv2.fillPoly(img, [np.array([[cx - half, sy], [cx - int(half * 1.9), sy + int(h * 0.06)],
                                 [cx - int(half * 1.7), int(h * 0.42)], [cx - half, int(h * 0.30)]], np.int32)],
                 (60, 60, 205), cv2.LINE_8)
    cv2.fillPoly(img, [np.array([[cx + half, sy], [cx + int(half * 1.9), sy + int(h * 0.06)],
                                 [cx + int(half * 1.7), int(h * 0.42)], [cx + half, int(h * 0.30)]], np.int32)],
                 (60, 60, 205), cv2.LINE_8)
    cv2.circle(img, (cx, sy), int(half * 0.38), (250, 250, 250), -1)          # 领口
    # 右：下装（宽扁的一块）—— 长宽比约 0.5 -> 判为 bottom
    x0, y0 = int(w * 0.60), int(h * 0.30)
    cv2.rectangle(img, (x0, y0), (x0 + int(w * 0.28), y0 + int(h * 0.17)), (200, 120, 40), -1)
    cv2.rectangle(img, (x0 + int(w * 0.10), y0 + int(h * 0.17)),
                  (x0 + int(w * 0.18), y0 + int(h * 0.30)), (200, 120, 40), -1)
    cv2.rectangle(img, (x0 + int(w * 0.10), y0), (x0 + int(w * 0.18), y0 + int(h * 0.02)),
                  (250, 250, 250), -1)
    return img


# ============================================================================
# 10. 自检 / CLI（要求 6：独立可测试性）
# ============================================================================
def self_test(verbose: bool = True) -> int:
    """模块独立自检：合成样图跑完整流水线 + 3 类校验失败用例，返回 0/1。"""
    ok_all = True

    def _line(title):
        print("-" * 74)
        print(f"[自检] {title}")

    print("=" * 74)
    print("[自检] 用户上传样图模块 —— 独立自检")
    print("=" * 74)
    print("[自检] 配置: " + json.dumps(
        {"precision": "balanced", "preview": False, "retries": 1,
         "allowed_formats": ["jpg", "jpeg", "png", "webp", "bmp"]}, ensure_ascii=False))

    work = os.path.join(DEFAULT_TMP_DIR, "_selftest")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    cfg = UploadConfig(preview=False, tmp_dir=os.path.join(work, "tmp"))

    # ---------- 用例 1：正常上传（含格式校验 / 落临时目录） ----------
    _line("用例 1/4：正常上传（PNG 白底样图）")
    sample = make_sample_image()
    sample_path = os.path.join(work, "sample_whitebg.png")
    assert _imwrite_unicode(sample_path, sample), "样图写入失败"
    print(f"[自检] 合成样图: {sample_path}  ({sample.shape[1]}x{sample.shape[0]})")
    try:
        up = upload_image(sample_path, cfg)
        print(f"[自检] 上传成功: session={os.path.basename(up.session)} "
              f"size={up.size_bytes}B sha1={up.sha1[:10]}... {up.w}x{up.h}")
        for k in ("path", "original_name", "size_bytes", "sha1", "width", "height", "ext", "session"):
            assert k in up.to_dict(), f"UploadedImage 缺字段 {k}"
    except UploadError as e:
        print(f"[自检] FAIL 上传失败: {e}"); ok_all = False; up = None

    # ---------- 用例 2：提取流水线 + 数据结构完整性 ----------
    _line("用例 2/4：衣物提取 + 标准化数据结构")
    result = None
    if up is not None:
        result = extract_from_file(up.path, cfg, out_dir=os.path.join(up.session, "items"))
        if not result.ok:
            print(f"[自检] FAIL 提取失败: {result.error}"); ok_all = False
        else:
            print(f"[自检] 提取成功: backend={result.backend} 检出 {len(result.items)} 件 "
                  f"耗时 {result.elapsed_ms:.1f}ms")
            need = {"item_id", "category", "confidence", "bbox", "bbox_norm", "polygon",
                    "polygon_norm", "mask_png", "rgba_png", "attributes", "source"}
            need_bbox = {"x", "y", "w", "h"}
            for it in result.items:
                d = it.to_dict()
                miss = need - set(d)
                assert not miss, f"GarmentItem 缺字段: {miss}"
                assert set(d["bbox"]) == need_bbox, "bbox 字段不对"
                assert os.path.exists(d["mask_png"]), f"mask 未落盘: {d['mask_png']}"
                assert os.path.exists(d["rgba_png"]), f"rgba 未落盘: {d['rgba_png']}"
                print(f"        - {d['item_id']}  {d['category']:9s} conf={d['confidence']:.2f} "
                      f"bbox={d['bbox']} poly={len(d['polygon'])}点")
            if len(result.items) != 2:
                print(f"[自检] FAIL 期望检出 2 件，实际 {len(result.items)} 件"); ok_all = False
            else:
                cats = sorted(it.category.value for it in result.items)
                print(f"[自检] 类别判定: {cats}（启发式；上装应含 'top'）")
                if "top" not in cats:
                    print("[自检] FAIL 未判定出上装"); ok_all = False
            # 下游对接：把抠图读回成 BGRA
            bgra = load_item_bgra(result.items[0])
            assert bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4, "抠图不是 BGRA"
            print(f"[自检] 下游对接: 抠图可读为 BGRA {bgra.shape[2]} 通道 "
                  f"({bgra.shape[1]}x{bgra.shape[0]})")

    # ---------- 用例 3：三类校验失败 + 异常码 ----------
    _line("用例 3/4：异常处理（格式非法 / 超大 / 损坏 / 过小）")
    bad_dir = os.path.join(work, "bad")
    os.makedirs(bad_dir, exist_ok=True)

    p_txt = os.path.join(bad_dir, "not_image.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write("hello")
    p_rot = os.path.join(bad_dir, "corrupted.png")
    with open(p_rot, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + os.urandom(512))
    p_big = os.path.join(bad_dir, "huge.png")
    with open(p_big, "wb") as f:
        f.write(b"\x00" * (2 * 1024 * 1024))
    p_tiny = os.path.join(bad_dir, "tiny.png")
    _imwrite_unicode(p_tiny, np.full((10, 10, 3), 255, np.uint8))

    cases = [
        ("格式非法(.txt)", lambda: validate_image(p_txt, cfg), "E_FORMAT"),
        ("文件损坏/非图片", lambda: validate_image(p_rot, cfg), "E_DECODE"),
        ("超大文件(2MB>上限)", lambda: validate_image(p_big, UploadConfig(preview=False, max_file_mb=0.5, tmp_dir=cfg.tmp_dir)), "E_SIZE"),
        ("尺寸过小(10x10)", lambda: validate_image(p_tiny, cfg), "E_DIMENSION"),
        ("文件不存在", lambda: validate_image(os.path.join(bad_dir, "nope.png"), cfg), "E_NOT_FOUND"),
    ]
    for label, fn, want in cases:
        try:
            fn()
            print(f"[自检] FAIL {label}: 未抛错（期望 {want}）"); ok_all = False
        except UploadError as e:
            good = (e.code == want)
            print(f"[自检] {'PASS' if good else 'FAIL'} {label}: {e.code} - {e.message}")
            ok_all &= good
        except Exception as e:
            print(f"[自检] FAIL {label}: 抛了非 UploadError 异常 {type(e).__name__}: {e}")
            ok_all = False

    # 提取越界：allowed_categories 只留 outerwear -> 应该报 E_NO_GARMENT
    try:
        r2 = extract_from_file(sample_path, UploadConfig(preview=False, tmp_dir=cfg.tmp_dir,
                                                        allowed_categories=("outerwear",),
                                                        retries=0))
        if r2.ok:
            print("[自检] FAIL 类别范围过滤未生效"); ok_all = False
        else:
            good = r2.error["error_code"] == "E_NO_GARMENT"
            print(f"[自检] {'PASS' if good else 'FAIL'} 类型范围过滤(allowed_categories=outerwear): "
                  f"{r2.error['error_code']} - {r2.error['error']}")
            ok_all &= good
    except Exception as e:
        print(f"[自检] FAIL 类型范围过滤用例异常: {e}"); ok_all = False

    # 后端注册表：未注册后端应给 E_BACKEND；AI 桩应给 E_BACKEND_UNAVAILABLE
    for label, be, want in [("未注册后端", "no_such_backend", "E_BACKEND"),
                            ("AI 桩未接入", "ai_stub", "E_BACKEND_UNAVAILABLE")]:
        r3 = extract_from_file(sample_path, UploadConfig(preview=False, tmp_dir=cfg.tmp_dir, retries=0), backend=be)
        good = (not r3.ok) and r3.error["error_code"] == want
        print(f"[自检] {'PASS' if good else 'FAIL'} {label}: "
              f"{(r3.error or {}).get('error_code')} - {(r3.error or {}).get('error')}")
        ok_all &= good

    # 临时存储：session 目录 + cleanup
    _line("用例 4/4：临时存储与过期清理")
    sessions = TempStore(cfg.tmp_dir).sessions()
    print(f"[自检] 临时 session 数: {len(sessions)}")
    if up is not None:
        has_upload = os.path.exists(os.path.join(up.session, "upload.json"))
        has_items = os.path.isdir(os.path.join(up.session, "items")) and \
            len(os.listdir(os.path.join(up.session, "items"))) > 0
        print(f"[自检] session 内产物: upload.json={has_upload} items/*={has_items}")
        ok_all &= has_upload and has_items
    removed = TempStore(cfg.tmp_dir, ttl_hours=0.0).cleanup()
    print(f"[自检] cleanup(ttl=0) 回收 session: {len(removed)} 个（模拟过期回收）")
    ok_all &= len(removed) >= 1

    # ---------- 证据图 ----------
    if result is not None and result.ok:
        vis = render_visualization(sample, result)
        vis_path = os.path.join(BASE_DIR, "upload_selftest_sheet.jpg")
        _imwrite_unicode(vis_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[自检] 可视化证据图已保存: {vis_path}")
        print("[自检] 标准化 JSON 片段: " + json.dumps(result.to_dict()["items"][0],
                                                        ensure_ascii=False)[:220] + " ...")

    print("=" * 74)
    print(f"[自检] 结论: {'PASS —— 上传/校验/提取/数据契约/异常/临时存储 全部通过' if ok_all else 'FAIL'}")
    print("=" * 74)
    return 0 if ok_all else 1


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用户上传样图 + 衣物提取模块（可独立运行）")
    ap.add_argument("--self-test", action="store_true", help="独立自检：合成样图跑完整流水线 + 失败用例")
    ap.add_argument("--image", type=str, default=None, help="对一张图片提取衣物并打印标准化 JSON")
    ap.add_argument("--out", type=str, default=None, help="可视化结果保存路径（默认 upload_result.jpg）")
    ap.add_argument("--backend", type=str, default=None, help="提取后端名（默认 white_bg）")
    ap.add_argument("--precision", type=str, default="balanced", choices=["fast", "balanced", "high"],
                    help="识别精度档位")
    ap.add_argument("--categories", type=str, default=None,
                    help="支持的衣物类型范围，逗号分隔，如 top,outerwear")
    ap.add_argument("--no-preview", action="store_true", help="关闭上传前预览")
    ap.add_argument("--list-backends", action="store_true", help="列出可用提取后端")
    ap.add_argument("--cleanup", action="store_true", help="清理过期的临时 session 并退出")
    args = ap.parse_args(argv)

    if args.list_backends:
        for e in list_extractors():
            print(f"  {'OK ' if e['available'] else 'N/A'}  {e['name']}")
        return 0

    if args.cleanup:
        removed = TempStore().cleanup()
        print(f"[清理] 回收过期 session {len(removed)} 个:")
        for d in removed:
            print("   -", d)
        return 0

    if args.self_test or not args.image:
        return self_test()

    cats = tuple(x.strip() for x in args.categories.split(",")) if args.categories else \
        UploadConfig.allowed_categories
    cfg = UploadConfig(preview=False if args.no_preview else True,
                       precision=args.precision, allowed_categories=cats)
    path = args.image
    if not os.path.exists(path):
        print(f"[错误] 图片不存在: {path}"); return 2
    try:
        up = upload_image(path, cfg)
    except UploadError as e:
        print(f"[错误] 上传失败: {e.code} - {e.message}"); return 1
    res = extract_from_file(up.path, cfg, backend=args.backend,
                            out_dir=os.path.join(up.session, "items"))
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    if not res.ok:
        print(f"[提示] {ERROR_CODES.get(res.error['error_code'], '')}")
        return 1
    out = args.out or os.path.join(BASE_DIR, "upload_result.jpg")
    _imwrite_unicode(out, render_visualization(_imread_unicode(up.path, cv2.IMREAD_COLOR), res),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[结果] 可视化已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
