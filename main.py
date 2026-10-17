# -*- coding: utf-8 -*-
"""
摄像头实时换衣 —— 最小可演示原型（CPU only）
================================================
流程：
  摄像头取帧 -> 水平镜像 -> MediaPipe Pose 取关键点(11/12 肩, 23/24 髋, 0 鼻)
  -> 由左右肩算：肩中心 / 肩宽 / 肩线倾角
  -> 把 assets/ 下的透明 PNG 上衣按"肩线倾角"旋转、按"肩宽*scale"缩放，
     锚点对齐到双肩中点，再做 alpha 混合叠到画面上。

按键：
  1 / 2 / 3  切换前 3 件（占位上衣）
  n / p      衣柜内前后切换（含 12 件从上传拼图切出的真实上衣）
  0          隐藏衣服
  r          重新扫描 / 校验衣柜（已缓存且未改动的文件不会重复解码）
  h          开关底部缩略图条（HUD）
  d          开关骨架调试画线
  u          上传一张样图 -> 自动提取衣物 -> 追加进衣柜（仅内存，见 docs/UPLOAD_MODULE.md）
  q / ESC    退出

资源目录（"衣服图片资源"按用途分类归档）：
  assets/clothes/placeholder/   代码生成的 3 件占位上衣
  assets/clothes/integrated/    从上传穿搭拼图切出的 12 件上衣（带透明通道）
  assets/refs/                  上传的原始参考图归档（人像 / 拼图，不参与渲染）
  assets/pose_landmarker_lite.task  姿态模型

限制（原型定位）：只处理正面、单人；不做遮挡、褶皱、形体变形；侧面/背面不保证贴合。

运行：python main.py                 （默认摄像头 0）
     python main.py --size 1280x720  （自定义窗口分辨率）
     python main.py --probe          （只探测可用摄像头）
     python main.py --image x.jpg --out preview.jpg --clothes 3   （图片自检）
     python main.py --selftest       （全面自检：逐件校验+渲染+contact sheet+缓存/耗时）
     python main.py --upload 样图.png            （上传样图并提取衣物，打印标准化 JSON）
     python main.py --upload 样图.png --upload-add（提取后加入衣柜并进入摄像头模式）
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# 资源目录结构（"衣服图片资源"按用途分类归档）
#   assets/clothes/placeholder/   代码生成的 3 件占位上衣
#   assets/clothes/integrated/    从上传的穿搭拼图切出的 12 件上衣（带透明通道）
#   assets/refs/                  上传的原始参考图归档（人像 / 拼图，不参与渲染）
#   assets/pose_landmarker_lite.task  姿态模型
CLOTHES_DIR = os.path.join(ASSETS_DIR, "clothes")
PLACEHOLDER_DIR = os.path.join(CLOTHES_DIR, "placeholder")
INTEGRATED_DIR = os.path.join(CLOTHES_DIR, "integrated")
REFS_DIR = os.path.join(ASSETS_DIR, "refs")

# mediapipe 1.x 的 Tasks API 需要这个模型文件（首次运行自动下载到 assets/）
POSE_MODEL_FILE = "pose_landmarker_lite.task"
POSE_MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
]


def _download_pose_model(path: str) -> None:
    """下载 MediaPipe Pose 的 .task 模型（仅 mediapipe 1.x 需要）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_err = None
    for url in POSE_MODEL_URLS:
        try:
            print(f"[模型] 正在下载姿态模型 -> {os.path.basename(path)} ...")
            with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
            print(f"[模型] 下载完成 ({os.path.getsize(path) / 1e6:.1f} MB)")
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "姿态模型下载失败: %s\n"
        "可手动下载后放到 %s\n"
        "或改用自带模型的旧版: pip install \"mediapipe==0.10.35\"" % (last_err, path)
    )

# ----------------------------------------------------------------------------
# CONFIG —— 手调这里就够了
#   scale        : 衣服宽度 = 肩宽 * scale（上衣含袖总宽，一般 1.6~2.0）
#   offset_x     : 水平偏移，单位=肩宽（正=向右）
#   offset_y     : 垂直偏移，单位=肩宽（正=向下）
#   angle_offset : 叠加到肩线倾角上的额外角度（度），用于手纠正
#   alpha        : 衣服不透明度 0~1
#   anchor       : PNG 内的"肩线锚点"，归一化 (0~1)。(0.5, 0.28) 表示
#                  "水平居中、高度 28% 处"那条线要对齐到人的双肩中点。
#                  换成真实衣服 PNG 时，这里改成衣服自己的肩线位置即可。
# ----------------------------------------------------------------------------
CONFIG = {
    "camera_index": 0,
    "cam_width": 1280,          # 向摄像头申请的采集分辨率（后面统一 resize）
    "cam_height": 720,
    "view_width": 960,          # 窗口显示分辨率（可用 --size 覆盖）
    "view_height": 540,
    "mirror": True,             # 水平镜像（自拍视角）
    "vis_threshold": 0.5,       # 关键点可见性阈值
    "model_complexity": 1,      # MediaPipe Pose: 0=最快, 1=均衡, 2=最准
    "debug_draw": False,        # 启动时是否画骨架调试线（运行中按 d 可切换）
    "hud": True,                # 底部缩略图条（运行中按 h 可切换）

    # --- 第 1 组：代码生成的 3 件占位上衣（assets/clothes/placeholder/）---
    "clothes": [
        {
            "name": "黑色外套",
            "path": "assets/clothes/placeholder/jacket_black.png",
            "scale": 1.87, "offset_x": 0.00, "offset_y": -0.05,
            "angle_offset": 0.0, "alpha": 0.95, "anchor": (0.5, 0.0),
        },
        {
            "name": "红色外套",
            "path": "assets/clothes/placeholder/jacket_red.png",
            "scale": 1.87, "offset_x": 0.00, "offset_y": -0.05,
            "angle_offset": 0.0, "alpha": 0.95, "anchor": (0.5, 0.0),
        },
        {
            "name": "白色衬衫",
            "path": "assets/clothes/placeholder/shirt_white.png",
            "scale": 1.87, "offset_x": 0.00, "offset_y": -0.05,
            "angle_offset": 0.0, "alpha": 0.95, "anchor": (0.5, 0.0),
        },
    ],

    # --- 第 2 组：从上传拼图切出的 12 件上衣（assets/clothes/integrated/）---
    #   anchor=(0.5, y) 里的 y 是"这件上衣肩线在 PNG 高度上的比例"，
    #   是把 PNG 顶部对齐到双肩中点的关键；已按每件图各自量好（见 anchor_y）。
    "integrated_default": {
        "scale": 1.65, "offset_x": 0.00, "offset_y": -0.03,
        "angle_offset": 0.0, "alpha": 0.95,
    },
    "anchor_y": {
        "look_01.png": 0.133, "look_02.png": 0.106, "look_03.png": 0.241,
        "look_04.png": 0.081, "look_05.png": 0.053, "look_06.png": 0.099,
        "look_07.png": 0.068, "look_08.png": 0.155, "look_09.png": 0.047,
        "look_10.png": 0.140, "look_11.png": 0.125, "look_12.png": 0.112,
    },
    # 想给某一件单独微调就写在这里（按文件名），字段可只写要改的：
    #   "tune": {"look_07.png": {"scale": 1.8, "offset_y": -0.06}}
    "tune": {},
}

# MediaPipe Pose 关键点编号
LM_NOSE = 0
LM_L_SHOULDER = 11
LM_R_SHOULDER = 12
LM_L_HIP = 23
LM_R_HIP = 24


# ============================================================================
# 1. 占位衣服 PNG 生成（assets 下没有真衣服图时用，保证一定能跑）
# ============================================================================
def _make_top_png(path: str, color_bgr, kind: str = "jacket") -> None:
    """用 OpenCV 画一件"上衣"剪影并存成带 alpha 通道的 PNG。

    kind: "jacket"（长袖、稍宽）/ "shirt"（短袖、稍窄）
    画法是"穿着态"剪影（袖子顺着胳膊往下垂），不是平铺图，所以叠到正面人像上更自然。
    画完会按 alpha 非零区域裁掉四周透明边，让 PNG 宽度 ≈ 衣服实际宽度，
    这样 CONFIG 里的 scale（衣服宽度 = 肩宽 * scale）才好估。
    """
    W, H = 600, 700
    cx = W // 2
    mask = np.zeros((H, W), np.uint8)

    shoulder_y = 200                       # 肩线所在高度（即 anchor 对齐位置）
    hem_y = 560                            # 下摆高度（大约落在髋部）
    if kind == "jacket":
        sleeve_out, sleeve_drop = 70, 215  # 长袖
    else:
        sleeve_out, sleeve_drop = 70, 165  # 短袖

    # --- 躯干（肩到下摆，略收腰）---
    half = 152
    body = np.array([
        [cx - half, shoulder_y], [cx - half + 6, 330], [cx - half + 12, 470], [cx - half, hem_y],
        [cx + half, hem_y], [cx + half - 12, 470], [cx + half - 6, 330], [cx + half, shoulder_y],
    ], np.int32)

    # --- 左右袖（从肩缝顺着胳膊往下垂）---
    left_sleeve = np.array([
        [cx - half, shoulder_y - 4],                      # 肩缝(内上)
        [cx - half - sleeve_out, shoulder_y + 56],        # 袖外上
        [cx - half - sleeve_out + 16, shoulder_y + sleeve_drop],   # 袖口外
        [cx - 166, shoulder_y + sleeve_drop - 30],       # 袖口内
    ], np.int32)
    right_sleeve = left_sleeve.copy()
    right_sleeve[:, 0] = 2 * cx - right_sleeve[:, 0]     # 关于中轴镜像

    cv2.fillPoly(mask, [body], 255, lineType=cv2.LINE_AA)
    cv2.fillPoly(mask, [left_sleeve], 255, lineType=cv2.LINE_AA)
    cv2.fillPoly(mask, [right_sleeve], 255, lineType=cv2.LINE_AA)

    # --- 挖领口（顶部中间挖出脖子洞）---
    cv2.ellipse(mask, (cx, shoulder_y), (58, 44), 0, 0, 360, 0, -1, cv2.LINE_AA)

    # --- 上色 + 深色细节（领口描边 / 下摆线）---
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = np.array(color_bgr, np.uint8)
    dark = (np.array(color_bgr, np.float32) * 0.62).astype(np.uint8)

    accent = np.zeros((H, W), np.uint8)
    cv2.ellipse(accent, (cx, shoulder_y), (58, 44), 0, 0, 360, 255, 11, cv2.LINE_AA)   # 领口
    cv2.line(accent, (cx - half + 2, hem_y - 6), (cx + half - 2, hem_y - 6), 255, 9, cv2.LINE_AA)
    accent = cv2.bitwise_and(accent, mask)
    rgb[accent > 0] = dark

    # --- 裁掉四周透明边：让 PNG 尺寸 = 衣服实际尺寸 ---
    ys, xs = np.where(mask > 0)
    top, bot = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    rgb, alpha = rgb[top:bot + 1, left:right + 1], mask[top:bot + 1, left:right + 1]

    # 建议 scale：让躯干宽度 ≈ 1.27 倍肩宽（躯干在 PNG 里占 2*half/总宽）
    torso_ratio = (2.0 * half) / alpha.shape[1]
    suggest = 1.27 / torso_ratio

    cv2.imwrite(path, np.dstack([rgb, alpha]))
    print(f"[assets] 已生成占位衣服: {os.path.basename(path)} "
          f"({kind}, {alpha.shape[1]}x{alpha.shape[0]}, 建议 anchor=(0.5, 0.0), "
          f"scale≈{suggest:.2f})")




def ensure_assets(verbose: bool = True) -> None:
    """保证 3 件占位衣服 PNG 存在；不存在则生成占位图（黑外套 / 红外套 / 白衬衫）。

    写入 assets/clothes/placeholder/（不覆盖 assets/clothes/integrated/ 里的真实素材）。
    """
    os.makedirs(PLACEHOLDER_DIR, exist_ok=True)
    plans = [
        ("jacket_black.png", (34, 34, 38), "jacket"),
        ("jacket_red.png", (62, 62, 200), "jacket"),
        ("shirt_white.png", (246, 246, 248), "shirt"),
    ]
    for fn, color, kind in plans:
        p = os.path.join(PLACEHOLDER_DIR, fn)
        if not os.path.exists(p):
            _make_top_png(p, color, kind)
        elif verbose:
            print(f"[assets] 已存在: {os.path.relpath(p, BASE_DIR)}")


# ============================================================================
# 2. 姿态 / 几何
# ============================================================================
def _create_pose(cfg: dict):
    """创建 MediaPipe Pose；兼容 mediapipe 0.10.x(solutions) 与 1.x(tasks) 两套 API。

    0.10.x：mp.solutions.pose.Pose，模型内置在 wheel 里。
    1.x  ：Tasks API，需要 pose_landmarker_lite.task（不存在会自动下载）。
    """
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(cfg["model_complexity"]),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        drawer = mp.solutions.drawing_utils
        return pose, drawer, "solutions"

    # ---- mediapipe 1.x：Tasks API ----
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "当前 mediapipe 既不提供 solutions API，也拿不到 tasks API。\n"
            '请执行: pip install "mediapipe==0.10.35"\n原始错误: %s' % e
        )

    model_path = os.path.join(ASSETS_DIR, POSE_MODEL_FILE)
    if not os.path.exists(model_path):
        _download_pose_model(model_path)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts), None, "tasks"



class PoseRunner:
    """统一封装 solutions / tasks 两种 mediapipe 的调用差异。"""

    def __init__(self, cfg: dict):
        self.pose, self.drawer, self.mode = _create_pose(cfg)
        self._t_ms = 0

    def process(self, rgb: np.ndarray, t_sec: float):
        """返回长度为 33 的 landmarks 列表（含 x/y/visibility），未检出则返回 None。"""
        if self.mode == "solutions":
            res = self.pose.process(rgb)
            if not res.pose_landmarks:
                return None, res
            return res.pose_landmarks.landmark, res

        import mediapipe as mp

        # Tasks 的 VIDEO 模式要求时间戳严格递增，这里做一次兜底
        self._t_ms = max(self._t_ms + 1, int(t_sec * 1000))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.pose.detect_for_video(mp_img, self._t_ms)
        if not res.pose_landmarks:
            return None, res
        return res.pose_landmarks[0], res

    def draw_skeleton(self, frame: np.ndarray, res) -> None:
        """调试模式：把检测到的骨架画出来。"""
        if self.mode == "solutions" and self.drawer is not None:
            if getattr(res, "pose_landmarks", None):
                self.drawer.draw_landmarks(
                    frame, res.pose_landmarks, self.pose.POSE_CONNECTIONS)
            return
        # tasks 模式没有现成 drawer，自己连几条主要线
        if getattr(res, "pose_landmarks", None):
            h, w = frame.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in res.pose_landmarks[0]]
            bones = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15),
                     (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]
            for a, b in bones:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)


def shoulder_metrics(lms, w: int, h: int, vis_thr: float):
    """由左右肩算 肩中心 / 肩宽(px) / 肩线倾角(度, 屏幕坐标 y 向下)。

    返回 (肩中心, 肩宽, 倾角) ；若左右肩可见性不达标则返回 None。
    """
    ls, rs = lms[LM_L_SHOULDER], lms[LM_R_SHOULDER]
    if min(ls.visibility, rs.visibility) < vis_thr:
        return None

    l = np.array([ls.x * w, ls.y * h], np.float64)
    r = np.array([rs.x * w, rs.y * h], np.float64)

    center = (l + r) / 2.0
    width = float(np.linalg.norm(r - l))
    if width < 2.0:          # 畸变/误检保护
        return None

    dx, dy = float(r[0] - l[0]), float(r[1] - l[1])
    angle = math.degrees(math.atan2(dy, dx))     # 屏幕坐标：正角=顺时针
    if angle > 90.0:                              # 归一化到 (-90, 90]
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return center, width, angle


# 调试模式要重点标出的关键点：鼻(黄) / 双肩(橙) / 双髋(蓝)
_DEBUG_POINTS = ((LM_NOSE, (0, 255, 255)),
                 (LM_L_SHOULDER, (255, 128, 0)), (LM_R_SHOULDER, (255, 128, 0)),
                 (LM_L_HIP, (0, 0, 255)), (LM_R_HIP, (0, 0, 255)))


def draw_debug_overlay(pose_runner: PoseRunner, frame: np.ndarray, lms, res,
                       cfg: dict, center=None) -> None:
    """调试画线：骨架 + 鼻/肩/髋关键点 + 双肩中点十字线。"""
    pose_runner.draw_skeleton(frame, res)
    h, w = frame.shape[:2]
    if lms is not None:
        for idx, col in _DEBUG_POINTS:
            lm = lms[idx]
            if lm.visibility >= cfg["vis_threshold"]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, col, -1, cv2.LINE_AA)
    if center is not None:
        cv2.drawMarker(frame, (int(center[0]), int(center[1])),
                       (0, 255, 0), cv2.MARKER_CROSS, 18, 2)



# ============================================================================
# 3. 衣柜：扫描 / 校验 / 缓存
# ============================================================================
def _imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED):
    """读图；cv2.imread 在中文路径下会失败，用 imdecode 兜底。"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception:
        return None


def _make_thumb(png_bgra: np.ndarray, target_h: int = 128) -> np.ndarray:
    """生成 HUD 用缩略图（等比缩放到高 target_h，合成到浅灰底上）。"""
    h, w = png_bgra.shape[:2]
    s = target_h / float(h)
    t = cv2.resize(png_bgra, (max(1, int(round(w * s))), target_h),
                   interpolation=cv2.INTER_AREA)
    a = t[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full(t.shape[:2] + (3,), 236.0, np.float32)
    return (t[:, :, :3].astype(np.float32) * a + bg * (1.0 - a)).astype(np.uint8)


class Wardrobe:
    """衣柜：扫描 assets/clothes/ 下的上衣 PNG，逐件校验，并缓存"解码结果 + 预乘 alpha + 缩略图"。

    缓存策略（对应"避免重复加载造成的资源浪费"）：
      * 按绝对路径在进程内缓存，同一文件只解码/预乘一次；
      * reload() 复用已有缓存，只有文件 mtime 变了才重新解码；
      * 逐帧渲染直接用缓存好的预乘数组，不再每帧重算 alpha/255 与乘法。
    """

    def __init__(self, cfg: dict, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.slots: list = []      # 可渲染的衣服
        self.report: list = []     # [(相对路径, 是否通过, 说明)]
        self._cache: dict = {}     # 绝对路径 -> {"mtime": float, "px": {...}}
        self.reload()

    # ---------- 扫描 ----------
    def _plan(self) -> list:
        """按目录约定给出 [(相对路径, 组名, 显示名, 渲染参数)]。"""
        plan = []
        for c in self.cfg["clothes"]:
            plan.append((c["path"], "placeholder", c["name"], dict(c)))

        if os.path.isdir(INTEGRATED_DIR):
            for fn in sorted(os.listdir(INTEGRATED_DIR)):
                if not fn.lower().endswith(".png"):
                    continue
                p = dict(self.cfg["integrated_default"])
                p.update(self.cfg.get("tune", {}).get(fn, {}))
                p["anchor"] = (0.5, float(self.cfg["anchor_y"].get(fn, 0.10)))
                plan.append((os.path.join("assets", "clothes", "integrated", fn),
                             "integrated", os.path.splitext(fn)[0], p))
        return plan

    # ---------- 校验 + 缓存 ----------
    def _load_cached(self, path: str):
        if not os.path.exists(path):
            return False, "文件不存在", None
        mtime = os.path.getmtime(path)
        hit = self._cache.get(path)
        if hit is not None and hit["mtime"] == mtime:
            return True, "缓存命中（未重复解码）", hit["px"]

        png = _imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if png is None:
            return False, "解码失败（文件损坏 / 不是图片 / 路径不可读）", None
        if png.ndim != 3 or png.shape[2] != 4:
            return False, f"缺少透明通道，实际 shape={png.shape}（必须是带 alpha 的 PNG）", None
        h, w = png.shape[:2]
        if min(h, w) < 32:
            return False, f"尺寸过小 {w}x{h}", None

        a255 = png[:, :, 3]
        cov = float((a255 > 8).mean())
        if cov < 0.02:
            return False, f"非透明像素仅占 {cov:.1%}（几乎全透明）", None

        a = a255.astype(np.float32) / 255.0
        px = {
            "w": w, "h": h, "src_a": a,
            "src_rgb": png[:, :, :3].astype(np.float32) / 255.0 * a[:, :, None],
            "thumb": _make_thumb(png),
        }
        self._cache[path] = {"mtime": mtime, "px": px}
        return True, f"OK {w}x{h}，不透明占比 {cov:.0%}", px

    def reload(self) -> None:
        """（重新）扫描并装配衣柜；已缓存且未改动的文件不会重新解码。"""
        self.slots, self.report = [], []
        for rel, group, name, params in self._plan():
            path = os.path.join(BASE_DIR, rel)
            ok, why, px = self._load_cached(path)
            self.report.append((rel, ok, why))
            if not ok:
                continue
            slot = dict(params)
            slot.update({"name": name, "path": path, "rel": rel, "group": group})
            slot.update(px)
            self.slots.append(slot)

    # ---------- 动态注入（"上传样图 -> 提取衣物 -> 直接穿上"的对接点）----------
    def add_slot_from_bgra(self, bgra: np.ndarray, name: str, params: dict,
                           rel: str = "<upload>") -> dict | None:
        """把一张 BGRA 数组当成新衣服加入衣柜（**只进内存**，不写 assets/clothes/）。

        复用与 `_load_cached()` 完全相同的缓存字段（src_a / src_rgb / thumb），
        所以 `overlay_item()` 可以像对待普通衣柜条目一样直接渲染它——
        这就是"提取结果无缝接入现有叠加业务"的接口。
        校验条件与静态资源一致（4 通道 / 尺寸 >=32 / 不透明占比 >=2%），不合格返回 None。
        """
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            return None
        h, w = bgra.shape[:2]
        if min(h, w) < 32:
            return None
        a255 = bgra[:, :, 3]
        cov = float((a255 > 8).mean())
        if cov < 0.02:
            return None
        a = a255.astype(np.float32) / 255.0
        px = {
            "w": w, "h": h, "src_a": a,
            "src_rgb": bgra[:, :, :3].astype(np.float32) / 255.0 * a[:, :, None],
            "thumb": _make_thumb(bgra),
        }
        slot = dict(params)
        slot.update({"name": name, "path": rel, "rel": rel, "group": "upload"})
        slot.update(px)
        self.slots.append(slot)
        self.report.append((rel, True, f"上传提取 OK {w}x{h}，不透明占比 {cov:.0%}"))
        return slot

    # ---------- 报告 ----------
    def print_report(self) -> int:
        """打印加载/校验表，返回 FAIL 的数量。"""
        bad = sum(1 for _r, ok, _w in self.report if not ok)
        print(f"[衣柜] 扫描 {len(self.report)} 个文件 -> 可用 {len(self.slots)} 件"
              f"（FAIL {bad} 个）")
        if self.verbose:
            for rel, ok, why in self.report:
                print(f"   {'PASS' if ok else 'FAIL'}  {rel:<46} {why}")
        return bad


# ============================================================================
# 4. 透明 PNG 叠加
# ============================================================================
def overlay_item(frame: np.ndarray, item: dict, center, shoulder_w: float,
                 angle_deg: float) -> None:
    """把 item（衣柜里的一件衣服）缩放/旋转/平移后 alpha 混合到 frame（原地修改 frame）。

    变换: p_out = s * R(phi) * (p_in - anchor) + target
      anchor : PNG 内归一化锚点 -> 像素（该点对齐到双肩中点）
      target : 双肩中点 + 偏移
    等比缩放（sx = sy）所以图片不会被拉变形；warpAffine 的 BORDER_CONSTANT(0,0,0,0)
    自动完成越界裁剪。预乘 alpha 的数组来自缓存，逐帧不重算。
    """
    ph, pw = int(item["h"]), int(item["w"])
    s = max(1e-3, (shoulder_w * float(item["scale"])) / float(pw))

    ax, ay = item.get("anchor", (0.5, 0.5))
    anchor = np.array([ax * pw, ay * ph], np.float64)

    target = np.array([
        center[0] + float(item.get("offset_x", 0.0)) * shoulder_w,
        center[1] + float(item.get("offset_y", 0.0)) * shoulder_w,
    ], np.float64)

    phi = math.radians(angle_deg + float(item.get("angle_offset", 0.0)))
    c, sn = math.cos(phi), math.sin(phi)
    A = s * np.array([[c, -sn], [sn, c]], np.float64)
    t = target - A @ anchor
    M = np.array([[A[0, 0], A[0, 1], t[0]],
                  [A[1, 0], A[1, 1], t[1]]], np.float32)

    # 只处理衣服落地的那块 ROI：先把 PNG 四角投影出来求包围盒，省掉整帧 warp 的开销
    fh, fw = frame.shape[:2]
    corners = np.array([[0, 0, 1], [pw, 0, 1], [pw, ph, 1], [0, ph, 1]], np.float64).T
    proj = M.astype(np.float64) @ corners
    x0 = max(0, int(math.floor(proj[0].min())) - 2)
    y0 = max(0, int(math.floor(proj[1].min())) - 2)
    x1 = min(fw, int(math.ceil(proj[0].max())) + 3)
    y1 = min(fh, int(math.ceil(proj[1].max())) + 3)
    if x1 - x0 < 2 or y1 - y0 < 2:      # 完全在画面外
        return
    size = (x1 - x0, y1 - y0)
    M_roi = M.copy()
    M_roi[0, 2] -= x0
    M_roi[1, 2] -= y0

    # 预乘 alpha 再插值：避免缩放/旋转后衣服边缘出现黑边
    alpha = cv2.warpAffine(item["src_a"], M_roi, size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)[:, :, None]
    rgb = cv2.warpAffine(item["src_rgb"], M_roi, size, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    k = float(item.get("alpha", 1.0))
    alpha *= k
    rgb *= k

    roi = frame[y0:y1, x0:x1]
    out = roi.astype(np.float32) / 255.0
    out = out * (1.0 - alpha) + rgb            # rgb 已是预乘色
    roi[:] = np.clip(out * 255.0, 0, 255).astype(np.uint8)



# ============================================================================
# ============================================================================
# 5. HUD（自适应缩略图条）
# ============================================================================
def draw_hud(frame: np.ndarray, slots: list, current: int) -> None:
    """底部缩略图条。格子尺寸按窗口宽度算，所以不同分辨率/设备下观感一致。"""
    n = len(slots)
    if n == 0:
        return
    H, W = frame.shape[:2]
    gap = max(3, W // 160)
    rows = 1 if n <= 8 else 2
    per_row = (n + rows - 1) // rows
    cell = int((W - 2 * gap - per_row * gap) / per_row)
    cell = max(24, min(112, cell))
    cellh = int(cell * 1.12)
    panel_h = rows * cellh + (rows + 1) * gap
    y0 = max(0, H - panel_h)

    panel = frame[y0:H]
    cv2.addWeighted(panel, 0.32, np.zeros_like(panel), 0.0, 0.0, panel)   # 压暗做背景板

    for i, s in enumerate(slots):
        r, c = divmod(i, per_row)
        x = gap + c * (cell + gap)
        y = y0 + gap + r * (cellh + gap)
        th = s["thumb"]
        sc = min(cell / th.shape[1], cellh / th.shape[0])
        tw, thh = max(1, int(th.shape[1] * sc)), max(1, int(th.shape[0] * sc))
        tx, ty = x + (cell - tw) // 2, y + (cellh - thh) // 2
        if ty + thh > H or tx + tw > W:
            continue
        frame[ty:ty + thh, tx:tx + tw] = cv2.resize(th, (tw, thh))
        on = (i == current)
        cv2.rectangle(frame, (x, y), (x + cell, y + cellh),
                      (0, 255, 0) if on else (120, 120, 120), 2 if on else 1)

    label = slots[current]["name"] if 0 <= current < n else "隐藏（按 0 / n 恢复）"
    cv2.putText(frame, f"{label}  [{current + 1 if current >= 0 else 0}/{n}]",
                (gap, min(H - 2, y0 + panel_h - gap // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, min(0.7, W / 1400)),
                (0, 255, 255) if current >= 0 else (160, 160, 160), 1, cv2.LINE_AA)


# ============================================================================
# 6. 摄像头
# ============================================================================
def open_camera(index: int, width: int, height: int):
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)   # Windows 下通常比 MSMF 更快更稳
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None


def probe_cameras(max_index: int = 4) -> list:
    """探测 0..max_index-1 哪些索引能打开并读到帧。"""
    found = []
    for i in range(max_index):
        cap = open_camera(i, CONFIG["cam_width"], CONFIG["cam_height"])
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h))
            cap.release()
    return found


CAMERA_HELP = """
摄像头打开失败，请依次排查：
  1) 摄像头是否被别的程序占用（微信/钉钉/腾讯会议/浏览器网页会议），全部关掉再试；
  2) Windows 设置 -> 隐私和安全性 -> 相机：打开"相机访问"和"让桌面应用访问相机"；
  3) 笔记本是否有物理摄像头开关/遮蔽盖；
  4) 换一个索引试试：python main.py --camera 1
  5) 先跑 python main.py --probe 看哪个索引可用；
  6) 完全没摄像头也能验证算法：python main.py --image 你的照片.jpg --out preview.jpg
"""


# ============================================================================
# 5. 主流程
# ============================================================================
def run_camera(args, pose_runner: PoseRunner, wardrobe: "Wardrobe") -> int:
    cfg = CONFIG
    cap = open_camera(args.camera, cfg["cam_width"], cfg["cam_height"])
    if cap is None:
        print("[错误] 打不开摄像头 index=%d（已尝试 DSHOW/MSMF 两种后端）" % args.camera)
        print(CAMERA_HELP)
        return 2

    W, H = cfg["view_width"], cfg["view_height"]
    print(f"[相机] 已打开 index={args.camera}，采集 "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f" -> 显示 {W}x{H}")
    print("[按键] 1/2/3 换前三件  n/p 前后换  r 重载衣柜  h HUD  d 调试  u 上传样图  0 隐藏  q 退出")

    slots = wardrobe.slots
    if not slots:
        print("[错误] 衣柜里没有可用衣服，无法启动。")
        cap.release()
        return 2
    current = args.clothes if 0 <= args.clothes < len(slots) else 0
    debug = cfg["debug_draw"]
    hud = cfg["hud"]
    fps = 0.0
    t_prev = time.time()
    last_ok = None          # 最近一次有效的 (肩中心, 肩宽, 倾角)
    miss_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[警告] 读帧失败，可能是摄像头被抢占或断连。")
            break
        frame = cv2.resize(frame, (W, H))
        if cfg["mirror"]:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()
        lms, res = pose_runner.process(rgb, now)

        metrics = None
        if lms is not None:
            metrics = shoulder_metrics(lms, W, H, cfg["vis_threshold"])

        if metrics is None:
            miss_frames += 1
            metrics = last_ok        # 肩不可见 -> 保持上一帧位置，避免错位乱跳
        else:
            last_ok = metrics
            miss_frames = 0

        if debug:
            # 注意：metrics 是 (中点, 肩宽, 倾角)，这里只把中点传进去
            draw_debug_overlay(pose_runner, frame, lms, res, cfg,
                               center=metrics[0] if metrics else None)

        # 叠加衣服（用衣柜里缓存的预乘数组，逐帧不重新解码/预乘）
        if current >= 0 and metrics is not None:
            center, sw, ang = metrics
            overlay_item(frame, slots[current], center, sw, ang)
            # 锚点十字线（方便手调）
            if debug:
                cv2.drawMarker(frame, (int(center[0]), int(center[1])),
                               (0, 255, 0), cv2.MARKER_CROSS, 18, 2)

        # FPS
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        # 状态栏
        name = slots[current]["name"] if current >= 0 else "隐藏"
        cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS {fps:5.1f}  当前: {name}  "
                           f"[1/2/3]换衣 [n/p]前后 [r]重载 [h]HUD [0]隐藏 [d]调试 [u]上传 [q]退出",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        if miss_frames > 8:
            cv2.putText(frame, "未检测到双肩：请正对镜头、上半身入画、光线充足",
                        (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2, cv2.LINE_AA)

        # 底部缩略图条（尺寸随窗口宽度缩放 -> 不同屏幕观感一致）
        if hud and not debug:
            draw_hud(frame, slots, current)

        cv2.imshow("Virtual Try-On (CPU) - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('0'):
            current = -1
        elif key == ord('1'):
            current = 0 if len(slots) > 0 else -1
        elif key == ord('2'):
            current = 1 if len(slots) > 1 else -1
        elif key == ord('3'):
            current = 2 if len(slots) > 2 else -1
        elif key == ord('n'):
            current = (current + 1) % len(slots)
        elif key == ord('p'):
            current = (current - 1) % len(slots)
        elif key == ord('h'):
            hud = not hud
        elif key == ord('r'):
            wardrobe.reload()
            slots = wardrobe.slots
            wardrobe.print_report()
            if not slots:
                current = -1
            elif current >= len(slots):
                current = len(slots) - 1
        elif key == ord('d'):
            debug = not debug
        elif key == ord('u'):
            # 上传样图 -> 自动提取衣物 -> 追加进衣柜（仅内存），可立刻按 n/p 穿上
            up, res, err = _upload_and_extract(None, preview=True)
            _print_upload_result(up, res, err, verbose=False)
            if res is not None and res.ok:
                n_add = _add_extracted_to_wardrobe(wardrobe, res)
                if n_add > 0:
                    slots = wardrobe.slots
                    current = len(slots) - 1      # 直接跳到刚追加的那件
                    print(f"[上传] 已向衣柜追加 {n_add} 件（仅内存，未写 assets/clothes/），"
                          f"当前共 {len(slots)} 件。")

        if cv2.getWindowProperty("Virtual Try-On (CPU) - press q to quit",
                                 cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[退出] 已释放摄像头并关闭窗口。")
    return 0


def run_image(args, pose_runner: PoseRunner, wardrobe: "Wardrobe") -> int:
    """无摄像头自检：对一张静态图跑完整流程并保存结果。"""
    # cv2.imread 不支持中文路径，用 imdecode 兜底
    try:
        img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[错误] 读不到图片: {args.image} ({e})")
        return 2
    if img is None:
        print(f"[错误] 读不到图片（路径或格式不对）: {args.image}")
        return 2
    slots = wardrobe.slots
    if not slots:
        print("[错误] 衣柜里没有可用衣服。")
        return 2
    idx = args.clothes
    if not (0 <= idx < len(slots)):
        print(f"[警告] --clothes={idx} 越界，已回退到 0（可用 0~{len(slots) - 1}）")
        idx = 0

    h, w = img.shape[:2]
    scale = min(CONFIG["view_width"] / w, CONFIG["view_height"] / h)
    frame = cv2.resize(img, (int(w * scale), int(h * scale)))

    lms, res = pose_runner.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time.time())
    metrics = shoulder_metrics(lms, frame.shape[1], frame.shape[0], CONFIG["vis_threshold"]) \
        if lms is not None else None
    if CONFIG["debug_draw"]:
        draw_debug_overlay(pose_runner, frame, lms, res, CONFIG,
                           center=metrics[0] if metrics else None)
    if metrics is None:
        print("[结果] 未检测到双肩（该图可能不是正面人像，或人物太小/太暗）")
        return 1

    center, sw, ang = metrics
    overlay_item(frame, slots[idx], center, sw, ang)
    print(f"[结果] 检出双肩: 中点=({center[0]:.0f},{center[1]:.0f}) 肩宽={sw:.0f}px 倾角={ang:.1f}度")
    out = args.out or os.path.join(BASE_DIR, "preview.jpg")
    # 同样用 imencode 落盘，兼容中文路径
    cv2.imencode(os.path.splitext(out)[1] or ".jpg", frame)[1].tofile(out)
    print(f"[结果] 已叠加「{slots[idx]['name']}」并保存: {out}")
    return 0


def run_selftest(args, pose_runner: PoseRunner, wardrobe: "Wardrobe") -> int:
    """全面自检：加载校验 + 逐件渲染 + contact sheet + 缓存命中 + 耗时统计。

    作为"要求 5：所有衣服图片均可正常访问、无加载失败、无显示异常"的可验证证据。
    返回 0 = 全部通过；1 = 存在 FAIL。
    """
    print("=" * 76)
    print("[自检] 步骤 1/5：资源加载与校验")
    print("=" * 76)
    fails = wardrobe.print_report()
    slots = wardrobe.slots
    if not slots:
        print("[自检] 结果: FAIL —— 衣柜为空，无法继续。")
        return 1

    # ---------- 2) 选一张"人形"输入帧 ----------
    print("=" * 76)
    print("[自检] 步骤 2/5：准备输入帧（用于验证渲染）")
    print("=" * 76)
    ref = os.path.join(REFS_DIR, "ref_person.jpg")
    base, metrics, src_note = None, None, ""
    if os.path.exists(ref):
        img = _imread_unicode(ref, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            sc = min(CONFIG["view_width"] / w, CONFIG["view_height"] / h)
            cand = cv2.resize(img, (int(w * sc), int(h * sc)))
            ch, cw = cand.shape[:2]
            lms, _res = pose_runner.process(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB), time.time())
            m = shoulder_metrics(lms, cw, ch, CONFIG["vis_threshold"]) if lms is not None else None
            if m is not None:
                base, metrics = cand, m
                src_note = f"参考人像 {os.path.basename(ref)}（真实检出双肩）"
    if base is None:
        W, H = CONFIG["view_width"], CONFIG["view_height"]
        base = np.full((H, W, 3), 226, np.uint8)
        # 画一个极简人体轮廓，便于肉眼判断"衣服该落在哪"
        cx, sh_y, hip_y = W // 2, int(H * 0.36), int(H * 0.78)
        half = int(W * 0.085)
        cv2.rectangle(base, (cx - half, sh_y), (cx + half, hip_y), (196, 196, 200), -1)
        cv2.circle(base, (cx, int(H * 0.24)), int(W * 0.055), (206, 206, 210), -1)
        cv2.line(base, (cx - half, sh_y), (cx - int(W * 0.22), int(H * 0.72)),
                 (196, 196, 200), max(3, W // 190))
        cv2.line(base, (cx + half, sh_y), (cx + int(W * 0.22), int(H * 0.72)),
                 (196, 196, 200), max(3, W // 190))
        metrics = (np.array([cx, float(sh_y)], np.float64), float(2 * half), 0.0)
        src_note = "合成参考帧（参考人像未检出双肩，改用几何约束）"
    center, sw, ang = metrics
    print(f"[自检] 输入帧: {base.shape[1]}x{base.shape[0]}  {src_note}")
    print(f"[自检] 双肩: 中点=({center[0]:.0f},{center[1]:.0f}) 肩宽={sw:.0f}px 倾角={ang:.1f}度")

    # ---------- 3) 逐件渲染 + 统计"是否真的画上去了" ----------
    print("=" * 76)
    print("[自检] 步骤 3/5：逐件渲染（检查是否有加载失败 / 显示异常）")
    print("=" * 76)
    rows = []
    tiles = []
    for i, s in enumerate(slots):
        frame = base.copy()
        overlay_item(frame, s, center, sw, ang)
        diff = cv2.absdiff(frame, base).max(axis=2)
        ys, xs = np.where(diff > 8)
        drawn = int(ys.size)
        ok = drawn > 0
        rows.append((s["rel"], drawn, ok))
        # 取"真正变化"的区域做缩略格（等比装进固定格子，不拉伸变形）
        if ok:
            x0, x1 = max(0, int(xs.min()) - 6), min(frame.shape[1], int(xs.max()) + 7)
            y0, y1 = max(0, int(ys.min()) - 6), min(frame.shape[0], int(ys.max()) + 7)
            crop = frame[y0:y1, x0:x1]
        else:
            crop = frame
        tiles.append((i, s["name"], crop, ok))

    # ---------- 4) 缓存命中 + 渲染耗时 ----------
    print("=" * 76)
    print("[自检] 步骤 4/5：缓存命中与渲染耗时")
    print("=" * 76)
    n_before = len(wardrobe._cache)
    wardrobe.reload()
    n_after = len(wardrobe._cache)
    cache_ok = (n_after == n_before) and (len(wardrobe.slots) == len(slots))
    print(f"[自检] 二次 reload: 缓存条目 {n_before} -> {n_after}，"
          f"装配 {len(wardrobe.slots)} 件  =>  {'未重复解码（命中）' if cache_ok else '出现重复解码'}")

    frames = [base.copy() for _ in slots]
    reps = 40
    t0 = time.perf_counter()
    for _ in range(reps):
        for s, f in zip(slots, frames):
            overlay_item(f, s, center, sw, ang)
    ms = (time.perf_counter() - t0) * 1000.0 / (reps * len(slots))
    print(f"[自检] 缓存后单件叠加耗时: {ms:.2f} ms/件（{reps} 轮 x {len(slots)} 件）")

    # ---------- 5) 拼 contact sheet 落盘 ----------
    print("=" * 76)
    print("[自检] 步骤 5/5：生成 contact sheet")
    print("=" * 76)
    TW, TH = 208, 244
    cols = 5
    nrow = (len(tiles) + cols - 1) // cols
    pad = 8
    LBL = 26
    sheet = np.full((nrow * (TH + LBL + pad) + pad, cols * (TW + pad) + pad, 3), 30, np.uint8)

    def _fit(im, tw, th):
        hh, ww = im.shape[:2]
        sc = min(tw / float(ww), th / float(hh))
        nw, nh = max(1, int(round(ww * sc))), max(1, int(round(hh * sc)))
        r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LANCZOS4)
        cv = np.full((th, tw, 3), 245, np.uint8)
        oy, ox = (th - nh) // 2, (tw - nw) // 2
        cv[oy:oy + nh, ox:ox + nw] = r
        return cv

    for i, name, crop, _ok in tiles:
        r, c = divmod(i, cols)
        x = pad + c * (TW + pad)
        y = pad + r * (TH + LBL + pad)
        sheet[y:y + TH, x:x + TW] = _fit(crop, TW, TH)
        cv2.rectangle(sheet, (x, y), (x + TW - 1, y + TH - 1), (90, 90, 90), 1)
        col = (0, 255, 0) if _ok else (0, 0, 255)
        cv2.putText(sheet, f"{i:02d} {name}", (x + 2, y + TH + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    sheet_path = os.path.join(BASE_DIR, "selftest_contact_sheet.jpg")
    cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(sheet_path)
    print(f"[自检] contact sheet 已保存: {sheet_path}（{len(tiles)} 格）")

    # ---------- 汇总 ----------
    bad_load = sum(1 for _r, ok, _w in wardrobe.report if not ok)
    bad_draw = sum(1 for _r, _d, ok in rows if not ok)
    print("=" * 76)
    print("[自检] 逐件渲染结果：")
    for rel, drawn, ok in rows:
        print(f"   {'PASS' if ok else 'FAIL'}  {rel:<46} 绘制像素 {drawn:>7}"
              f"  （约占画面 {drawn / float(base.shape[0] * base.shape[1]):.1%}）")
    print("-" * 76)
    print(f"[自检] 汇总: 加载失败 {bad_load} 件 / 渲染失败 {bad_draw} 件 / "
          f"缓存命中 {'是' if cache_ok else '否'} / 单件 {ms:.2f} ms")
    total_fail = fails + bad_draw + (0 if cache_ok else 1)
    print(f"[自检] 结论: {'PASS —— 所有衣服图片均可访问、无加载失败、无显示异常' if total_fail == 0 else f'FAIL（{total_fail} 项）'}")
    print("=" * 76)
    return 0 if total_fail == 0 else 1



def _upload_and_extract(path, preview: bool = True, backend: str | None = None):
    """调用 garment_upload 完成"上传 + 衣物提取"，把所有失败都转成结构化 error。

    返回 (UploadedImage|None, GarmentResult|None, error_dict|None)：
      * 上传失败（格式/大小/损坏/取消）-> uploaded=None, error 带 error_code
      * 提取失败                      -> uploaded 有值, result.ok=False, error 有值
    模块缺失时也返回结构化 error，保证 main.py 不因它崩掉。
    """
    try:
        from garment_upload import UploadConfig, upload_and_extract, UploadError
    except Exception as e:                       # 模块不可用也不影响原有功能
        return None, None, {"ok": False, "error_code": "E_MODULE",
                            "error": f"garment_upload 模块不可用: {e}"}
    cfg = UploadConfig(preview=bool(preview))
    try:
        up, res = upload_and_extract(path, cfg, backend=backend)
    except UploadError as e:
        return None, None, e.to_dict()
    return up, res, (None if res.ok else res.error)


def _print_upload_result(up, res, error, verbose: bool = True) -> None:
    """打印上传/提取结果。verbose=True 时输出完整标准化 JSON（CLI 用）。"""
    if up is not None:
        print(f"[上传] 已接收: {os.path.basename(up.path)}  {up.w}x{up.h}  "
              f"{up.size_bytes / 1024:.1f}KB  sha1={up.sha1[:10]}...")
        print(f"[上传] 临时目录: {up.session}")
    if error is not None:
        hint = ""
        try:
            from garment_upload import ERROR_CODES
            hint = ERROR_CODES.get(error.get("error_code"), "")
        except Exception:
            pass
        print(f"[上传] 失败: {error.get('error_code')} - {error.get('error')}")
        if hint:
            print(f"[上传] 说明: {hint}")
        return
    if res is None:
        return
    print(f"[提取] 后端={res.backend}  图 {res.width}x{res.height}  "
          f"检出 {len(res.items)} 件  耗时 {res.elapsed_ms:.1f} ms")
    for it in res.items:
        b = it.bbox
        print(f"       - {it.category.value:9s} conf={it.confidence:.2f} "
              f"bbox=({b.x},{b.y},{b.w},{b.h}) 轮廓 {len(it.polygon)} 点  {it.rgba_png}")
    if verbose:
        import json
        print("[提取] 标准化输出（schema garment.v1）:")
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


def _add_extracted_to_wardrobe(wardrobe: "Wardrobe", res, prefix: str = "上传") -> int:
    """把提取到的衣物注入衣柜内存（"上传样图 -> 直接穿上"的演示闭环）。

    平铺抠图的肩线一般靠近顶部，所以 anchor 取 (0.5, 0.10)；
    其余渲染参数沿用 integrated_default。返回成功加入的件数。
    """
    base = dict(CONFIG["integrated_default"])
    base["anchor"] = (0.5, 0.10)
    added = 0
    for i, item in enumerate(res.items):
        bgra = _imread_unicode(item.rgba_png, cv2.IMREAD_UNCHANGED)
        if bgra is None:
            print(f"[上传] 跳过 {item.item_id}：抠图读取失败")
            continue
        name = f"{prefix}-{i + 1:02d}-{item.category.value}"
        if wardrobe.add_slot_from_bgra(bgra, name, dict(base)):
            added += 1
        else:
            print(f"[上传] 跳过 {item.item_id}：不满足衣柜条目校验")
    return added


def _parse_size(text: str):
    """解析 --size 参数，支持 '960x540' / '960,540' / '1280*720'。"""
    t = text.lower().replace("*", "x").replace(",", "x")
    if "x" not in t:
        raise argparse.ArgumentTypeError("--size 格式应为 WxH，例如 960x540")
    a, b = t.split("x", 1)
    try:
        w, h = int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError("--size 必须是整数，例如 960x540")
    if w < 160 or h < 120:
        raise argparse.ArgumentTypeError("--size 过小（最小 160x120）")
    return w, h


def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头实时换衣最小原型 (CPU)")
    ap.add_argument("--camera", type=int, default=CONFIG["camera_index"], help="摄像头索引，默认 0")
    ap.add_argument("--probe", action="store_true", help="只探测可用摄像头索引后退出")
    ap.add_argument("--image", type=str, default=None, help="用静态图片自检（无摄像头时）")
    ap.add_argument("--out", type=str, default=None, help="--image 模式的结果保存路径")
    ap.add_argument("--clothes", type=int, default=0,
                    help="使用衣柜里的第几件（0 起；--image 与摄像头模式都生效）")
    ap.add_argument("--size", type=_parse_size, default=None,
                    help="窗口显示分辨率，例如 960x540 / 1280x720（不同屏幕尺寸一致性）")
    ap.add_argument("--selftest", action="store_true",
                    help="全面自检：逐件校验+渲染+生成 contact sheet+缓存/耗时统计")
    ap.add_argument("--no-hud", action="store_true", help="关闭底部缩略图条")
    ap.add_argument("--debug", action="store_true", help="启动即开启骨架/锚点调试画线")
    # --- 用户上传样图模块（见 docs/UPLOAD_MODULE.md）---
    ap.add_argument("--upload", type=str, default=None,
                    help="上传一张样图并自动提取衣物，打印标准化 JSON（默认提取完就退出）")
    ap.add_argument("--upload-preview", dest="upload_preview", action="store_true", default=None,
                    help="上传前弹出预览窗口确认（默认开启；Enter 确认 / ESC 取消）")
    ap.add_argument("--no-upload-preview", dest="upload_preview", action="store_false",
                    help="跳过上传前预览（无 GUI / 批量场景）")
    ap.add_argument("--upload-backend", type=str, default=None,
                    help="提取后端名，默认 white_bg（可用列表见 garment_upload.list_extractors()）")
    ap.add_argument("--upload-add", action="store_true",
                    help="与 --upload 联用：把提取到的衣物加入衣柜（仅内存）后进入摄像头模式")
    args = ap.parse_args()

    if args.probe:
        found = probe_cameras()
        if found:
            for i, w, h in found:
                print(f"[探测] 摄像头 index={i} 可用，默认分辨率 {w}x{h}")
        else:
            print("[探测] 0~3 均不可用。")
            print(CAMERA_HELP)
        return 0

    ensure_assets()

    if args.size is not None:
        CONFIG["view_width"], CONFIG["view_height"] = args.size
    CONFIG["debug_draw"] = args.debug
    if args.no_hud:
        CONFIG["hud"] = False

    # 构建衣柜：扫描 assets/clothes/，逐件校验并建立缓存
    wardrobe = Wardrobe(CONFIG)
    if not args.selftest and not args.upload and wardrobe.print_report() > 0:
        print("[警告] 有文件未通过校验（详情见上表），它们不会进入衣柜。")

    # --- 上传样图：上传 -> 提取衣物 -> 标准化 JSON；--upload-add 时顺带注入衣柜 ---
    if args.upload:
        preview = True if args.upload_preview is None else bool(args.upload_preview)
        up, res, err = _upload_and_extract(args.upload, preview=preview, backend=args.upload_backend)
        _print_upload_result(up, res, err, verbose=True)
        if err is not None or res is None:
            return 1
        if not args.upload_add:
            return 0
        n_add = _add_extracted_to_wardrobe(wardrobe, res)
        if n_add <= 0:
            print("[上传] 没有任何衣物满足衣柜条目校验，无法加入衣柜。")
            return 1
        print(f"[上传] 已向衣柜追加 {n_add} 件（仅内存，未写 assets/clothes/）。"
              f"继续进入摄像头模式，按 n/p 翻到「上传-xx」即可穿上。")

    if args.selftest:
        try:
            pose_runner = PoseRunner(CONFIG)
        except Exception as e:
            print("[错误] 初始化 MediaPipe Pose 失败：%s" % e)
            return 3
        print(f"[姿态] MediaPipe 就绪（API: {pose_runner.mode}）")
        return run_selftest(args, pose_runner, wardrobe)

    if not wardrobe.slots:
        print("[错误] 衣柜为空（assets/clothes/ 下没有可用 PNG）。")
        return 2

    try:
        pose_runner = PoseRunner(CONFIG)
    except Exception as e:
        print("[错误] 初始化 MediaPipe Pose 失败：%s" % e)
        print("      可尝试: pip install -U mediapipe opencv-python numpy")
        return 3
    print(f"[姿态] MediaPipe 就绪（API: {pose_runner.mode}）")

    if args.image:
        return run_image(args, pose_runner, wardrobe)
    return run_camera(args, pose_runner, wardrobe)



if __name__ == "__main__":
    sys.exit(main())
