#!/usr/bin/env python
"""人脸门禁 Demo 主程序。

用法:
    python main.py recognize              实时门禁识别(默认命令)
    python main.py enroll --name 张三      录入人脸
    python main.py list                   查看人脸库
    python main.py remove --name 张三      删除某人(支持 ID 或姓名子串)
    python main.py clear                  清空人脸库
    python main.py selftest               环境自检 + 阈值标定
    python main.py cameras                列出可用摄像头序号
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=FutureWarning)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facegate.camera import list_cameras
from facegate.config import Config, dump_config, load_config
from facegate.engine import PROFILE_MINIMAL_VIZ, PROFILE_ORDER, FaceEngine, FaceObs
from facegate.gallery import DuplicateSample, Gallery
from facegate.gate import EventJournal, Gate, MetricsWriter
from facegate.liveness.analyzer import LivenessAnalyzer
from facegate.util import FpsMeter, imwrite_u, sanitize_id
from facegate.viz import BASE_H, BASE_W, Dashboard
from facegate.viz.panels import Canvas, MAIN_H, MAIN_W, draw_manage

WINDOW = "FaceGate Demo"
CUE_KEYS = ["deep_model", "blink", "micro_motion", "planarity", "texture", "moire", "color"]
# OpenCV 在不同后端下方向键的码值不一致,全部兼容
KEY_UP = (81, 65322, 38)
KEY_DOWN = (84, 65325, 40)


def _screen_size() -> tuple:
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
        return int(ctypes.windll.user32.GetSystemMetrics(0)), int(ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:  # noqa: BLE001
        return 1920, 1080


def _open_window() -> float:
    """创建可缩放窗口并按屏幕分辨率等比适配,返回缩放系数。"""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    keep = getattr(cv2, "WINDOW_KEEPRATIO", None)
    if keep is not None:
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_ASPECT_RATIO, keep)
        except cv2.error:
            pass
    sw, sh = _screen_size()
    scale = min(sw * 0.96 / BASE_W, sh * 0.94 / BASE_H)
    scale = float(max(0.55, min(scale, 2.2)))
    ww, wh = int(BASE_W * scale), int(BASE_H * scale)
    cv2.resizeWindow(WINDOW, ww, wh)
    cv2.moveWindow(WINDOW, max(0, (sw - ww) // 2), max(0, (sh - wh) // 2))
    return scale


def _ask_name(title: str, initial: str = "") -> str:
    """弹出系统输入框(支持中文输入法);无 Tk 环境时退回终端输入。"""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        val = simpledialog.askstring(title, "姓名或 ID:", initialvalue=initial, parent=root)
        root.destroy()
        return (val or "").strip()
    except Exception:  # noqa: BLE001
        try:
            return input(f"{title} — 姓名或 ID: ").strip()
        except EOFError:
            return ""


# ===========================================================================
# 公共:构建运行时对象
# ===========================================================================


def build(cfg: Config, need_camera: bool = True):
    engine = FaceEngine(cfg.engine)
    gallery = Gallery.open(cfg.gallery, cfg.gallery_dir)
    analyzer = LivenessAnalyzer(cfg.liveness)
    dash = Dashboard(cfg)
    src = None
    if need_camera:
        from facegate.camera import VideoSource

        src = VideoSource(cfg.camera)
    return engine, gallery, analyzer, dash, src


def help_text() -> str:
    return (
        "\n快捷键(识别模式)\n"
        "  q / ESC   退出            空格  暂停/继续\n"
        "  f         全屏 / 退出全屏(窗口也可任意拖拽伸缩,画面等比自适应)\n"
        "  p         循环切换性能档 quality / balanced / fast\n"
        "  e         弹框录入当前人脸   m  人脸库管理\n"
        "  r         重置活体轨迹       s  截图    v  切换叠加层    h  帮助\n"
        "  c         切换摄像头(或点工具栏「摄像头:n」;列表见 python main.py cameras)\n"
        "  1..7      单独开关七路活体线索\n"
        "\n鼠标(识别模式)\n"
        "  顶部工具栏按钮:暂停/录入/管理/截图/全屏/性能/叠加/重置/摄像头\n"
        "  点击「七路线索得分」某一行 = 开关该线索\n"
        "  点击「库内相似度」某一行   = 进入管理并选中该人\n"
        "  点击「载体类型匹配」某一行 = 显示该类的期望签名\n"
        "\n快捷键/鼠标(管理模式)\n"
        "  ↑/↓ 或 w/s 或点击行  选择人员\n"
        "  d 或「删除选中」  删除(两次确认)     r 或「改名」  弹框改名\n"
        "  c 或「清空全库」  清空(两次确认)     e 或「录入新人」  录入\n"
        "  m / ESC / 「返回识别」  回到识别模式\n"
    )


# ===========================================================================
# 识别
# ===========================================================================


def cmd_recognize(cfg: Config) -> int:
    engine, gallery, analyzer, dash, src = build(cfg)
    gate = Gate(cfg, gallery)
    fps = FpsMeter()
    dfps = FpsMeter()

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    metrics = MetricsWriter(cfg.log_dir / f"metrics_{stamp}.csv", cfg.log.metrics_csv)
    journal = EventJournal(cfg.log_dir / f"events_{stamp}.jsonl")

    print(f"人脸库: {cfg.gallery_dir} (已录入 {len(gallery)} 人 / {gallery.total_samples} 条特征)")
    print(f"指标日志: {metrics.path if metrics.enabled else '已关闭'}")
    print(help_text())

    embedding: Optional[np.ndarray] = None
    embed_req_at = 0.0
    embed_set_at = 0.0
    paused = False
    notice = ""
    notice_until = 0.0
    logged_events = 0
    last_frame = None
    image = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
    obs: Optional[FaceObs] = None
    lv = None
    mode = "live"
    sel = 0
    pending = ""
    pending_until = 0.0
    manage_notice = ""

    if cfg.viz.enabled:
        print(f"窗口缩放 {_open_window():.2f}x(基准 {BASE_W}x{BASE_H},按屏幕等比适配)")

    clicks: "deque" = deque()

    def _on_mouse(event, mx, my, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(mx), int(my)))

    fullscreen = False
    last_win = (0, 0)
    overlay_name = "全开"

    def _toggle_fullscreen() -> str:
        nonlocal fullscreen, last_win
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )
        if not fullscreen:
            last_win = (0, 0)
            _open_window()
        return "已全屏(再按 f 或点按钮退出)" if fullscreen else "已退出全屏"

    def _next_profile() -> str:
        order = PROFILE_ORDER
        nxt = order[(order.index(engine.profile_name) + 1) % len(order)]
        engine.set_profile(nxt)
        cfg.viz.minimal = bool(PROFILE_MINIMAL_VIZ.get(nxt, False))
        analyzer.reset("切换性能档")
        return (
            f"性能档 → {nxt}(det {engine.profile['det_size']}, ROI {'开' if engine.profile['roi'] else '关'}"
            f"{', 精简面板' if cfg.viz.minimal else ''})"
        )

    def _toggle_cue(k: str) -> str:
        spec = cfg.liveness.cues.get(k)
        if spec is None:
            return ""
        spec.enabled = not spec.enabled
        state = "开" if spec.enabled else "关"
        print(f"线索 {k} 已{state}启")
        return f"{k} → {state}"

    def _show(base: np.ndarray) -> None:
        """按当前窗口尺寸等比 letterbox 显示;窗口被拖拽改变时下一帧自动适配。

        注意必须先 imshow 再查窗口尺寸:首帧之前 W32 窗口尚未 realize,
        getWindowImageRect 会抛 NULL window。
        """
        nonlocal last_win
        ww, wh = last_win
        if ww < 64 or wh < 64:
            ww, wh = BASE_W, BASE_H
        cv2.imshow(WINDOW, dash.display(base, ww, wh))
        try:
            rect = cv2.getWindowImageRect(WINDOW)
            if rect[2] >= 64 and rect[3] >= 64:
                last_win = (int(rect[2]), int(rect[3]))
        except cv2.error:
            pass

    def _window_closed() -> bool:
        """用户点了窗口 X 按钮:OpenCV 不会抛异常,只会把窗口标为不可见。"""
        if not cfg.viz.enabled:
            return False
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _switch_camera() -> str:
        """切到下一个可用摄像头;换源后轨迹与特征必须清空,否则跨设备比较无意义。"""
        nonlocal embedding
        if src.is_file:
            return "视频文件源不支持切换摄像头"
        cams = [i for i, _ in list_cameras()]
        if not cams:
            return "没有探测到可用摄像头"
        others = [i for i in cams if i != src.index]
        if not others:
            return f"只探测到摄像头 {src.index}"
        try:
            label = src.switch_camera(others[0])
        except RuntimeError as e:
            return str(e)
        embedding = None
        analyzer.reset("切换摄像头")
        w, h = src.size
        msg = f"已切换到 {label}({w}×{h})"
        print(msg)
        return msg

    if cfg.viz.enabled:
        cv2.setMouseCallback(WINDOW, _on_mouse)

    def _do_delete(people) -> str:
        nonlocal sel, pending, pending_until
        if not people:
            return ""
        if pending == "delete":
            gone = gallery.remove(people[sel].id)
            pending = ""
            sel = max(0, sel - 1)
            msg = f"已删除:{', '.join(p.name for p in gone)}"
            print(msg)
            return msg
        pending, pending_until = "delete", time.time() + 3
        return f"待删除:{people[sel].name}(id={people[sel].id})"

    def _do_clear(people) -> str:
        nonlocal sel, pending, pending_until
        if not people:
            return ""
        if pending == "clear":
            n = gallery.clear()
            pending, sel = "", 0
            msg = f"已清空 {n} 人"
            print(msg)
            return msg
        pending, pending_until = "clear", time.time() + 3
        return f"待清空全部 {len(people)} 人"

    def _do_rename(people) -> str:
        if not people:
            return ""
        new = _ask_name("改名", people[sel].name)
        if new and gallery.rename(people[sel].id, new):
            return f"已改名为 {new}"
        return ""

    try:
        while True:
            # ---------------- 人脸库管理模式 ----------------
            if mode == "manage":
                src.read()                      # 保持视频流新鲜,但不跑推理
                people = sorted(gallery.people.values(), key=lambda p: p.created_at)
                sel = max(0, min(sel, len(people) - 1))
                if pending and time.time() > pending_until:
                    pending, manage_notice = "", ""
                cvs = Canvas(dash.painter, BASE_W, BASE_H)
                draw_manage(cvs, gallery, sel, pending, manage_notice)
                dash.hits = cvs.hits
                base = cvs.finish()
                if cfg.viz.enabled:
                    _show(base)
                    raw = cv2.waitKey(20)
                else:
                    raw = -1
                if _window_closed():
                    print("窗口已关闭。")
                    break
                key = raw & 0xFF
                act, pay = (None, None)
                while clicks:
                    mx, my = clicks.popleft()
                    act, pay = dash.hit_test(*dash.to_base(mx, my))
                    if act:
                        break

                if act == "mg_select":
                    sel, pending = int(pay), ""
                elif act == "mg_up":
                    sel = max(0, sel - 1)
                elif act == "mg_down":
                    sel = min(len(people) - 1, sel + 1)
                elif act == "mg_delete":
                    manage_notice = _do_delete(people)
                elif act == "mg_clear":
                    manage_notice = _do_clear(people)
                elif act == "mg_rename":
                    manage_notice = _do_rename(people)
                elif act == "mg_enroll":
                    _gui_enroll(cfg, engine, gallery, analyzer, dash, src)
                    mode, pending = "live", ""
                elif act == "mg_back":
                    mode, pending = "live", ""
                elif act == "fullscreen":
                    manage_notice = _toggle_fullscreen()
                elif act and pending:
                    pending, manage_notice = "", ""

                if raw == 27 or key == ord("m") or key == ord("q"):
                    mode, pending = "live", ""
                elif not people:
                    if key == ord("e"):
                        _gui_enroll(cfg, engine, gallery, analyzer, dash, src)
                        mode = "live"
                elif raw in KEY_UP or key == ord("w"):
                    sel = max(0, sel - 1)
                elif raw in KEY_DOWN or key == ord("s"):
                    sel = min(len(people) - 1, sel + 1)
                elif key == ord("d"):
                    manage_notice = _do_delete(people)
                elif key == ord("c"):
                    manage_notice = _do_clear(people)
                elif key == ord("r"):
                    manage_notice = _do_rename(people)
                elif key == ord("e"):
                    _gui_enroll(cfg, engine, gallery, analyzer, dash, src)
                    mode, pending = "live", ""
                continue

            # ---------------- 识别模式 ----------------
            frame = None if paused else src.read()
            if frame is None and not paused:
                print("视频源结束或摄像头断开。")
                break
            if frame is None:
                frame = last_frame          # 暂停时复用上一帧,画面保持可交互
                if frame is None:
                    break
            else:
                last_frame = frame
                image = frame.image

            if not paused:
                now = time.perf_counter()
                with_pose = (frame.index % max(1, cfg.engine.pose_every_n_frames)) == 0
                obs_list = engine.detect(image, with_pose=with_pose, frame_id=frame.index)
                obs = obs_list[0] if obs_list else None

                if obs is None:
                    analyzer.mark_lost()
                    embedding = None
                    embed_req_at = embed_set_at = 0.0
                    engine.reset_roi()
                else:
                    analyzer.push(obs, image, frame.index, cfg.engine.spectrum_every_n_frames)
                    got = engine.take_embedding()
                    if got is not None:
                        embedding = got
                        embed_set_at = now
                    # 慢路径异步化:ArcFace 提取在后台线程跑,主循环只取最近一次结果
                    if embedding is None or (now - embed_req_at) * 1000.0 >= cfg.engine.recognize_interval_ms:
                        if engine.request_embed(obs.aligned):
                            embed_req_at = now

            lv = (
                analyzer.evaluate()
                if analyzer.samples and (frame.index % max(1, int(engine.profile["eval_every"])) == 0 or lv is None)
                else lv
            )
            st = gate.update(obs is not None, lv, embedding, (time.perf_counter() - embed_set_at) * 1000.0)
            matches = gallery.match(embedding, top_k=max(1, len(gallery.people))) if embedding is not None else []

            if not paused:
                _write_metrics(metrics, frame, obs, lv, st, matches, analyzer)
            if len(gate.events) > logged_events:
                journal.append(gate.events[logged_events:])
                logged_events = len(gate.events)

            f_now = fps.tick()
            ui_state = {
                "paused": paused,
                "profile": engine.profile_name,
                "overlay": overlay_name,
                "fullscreen": fullscreen,
                "camera": None if src.is_file else src.index,
            }
            if cfg.viz.enabled:
                base = dash.render(
                    image, obs, st, lv, analyzer, matches, f_now, dfps.value, engine.stats, src.label,
                    similarity=matches[0].similarity if matches else 0.0,
                    gallery_total=len(gallery),
                    notice=("已暂停" if paused else notice) if (paused or time.time() < notice_until) else "",
                    ui_state=ui_state,
                )
                _show(base)
                dfps.tick()
                raw = cv2.waitKey(1)
            else:
                base = None
                raw = -1
                if frame.index % 20 == 0:
                    _print_status(frame, obs, lv, st, matches, f_now, engine.stats)
            if _window_closed():
                print("窗口已关闭。")
                break
            key = raw & 0xFF

            act, pay = (None, None)
            while clicks:
                mx, my = clicks.popleft()
                act, pay = dash.hit_test(*dash.to_base(mx, my))
                if act:
                    break

            if act == "pause":
                paused = not paused
            elif act == "enroll":
                _gui_enroll(cfg, engine, gallery, analyzer, dash, src)
                notice, notice_until = "录入流程结束,回到识别", time.time() + 3
                embedding = None
                analyzer.reset("录入后重置")
            elif act == "manage":
                mode, sel, pending = "manage", 0, ""
                manage_notice = ""
            elif act == "snapshot":
                if base is not None:
                    _snapshot(cfg, base)
                    notice, notice_until = "已保存截图", time.time() + 3
            elif act == "fullscreen":
                notice, notice_until = _toggle_fullscreen(), time.time() + 2
            elif act == "perf_next":
                notice, notice_until = _next_profile(), time.time() + 3
            elif act == "overlay_next":
                notice, overlay_name, notice_until = *_cycle_overlay(cfg), time.time() + 2
            elif act == "reset_track":
                analyzer.reset("手动重置")
                embedding = None
                notice, notice_until = "轨迹已重置", time.time() + 2
            elif act == "camera_next":
                notice, notice_until = _switch_camera(), time.time() + 3
            elif act == "toggle_cue":
                notice, notice_until = _toggle_cue(pay), time.time() + 2
            elif act == "gallery_select":
                ids = [p.id for p in sorted(gallery.people.values(), key=lambda p: p.created_at)]
                mode, sel, pending = "manage", (ids.index(pay) if pay in ids else 0), ""
                manage_notice = ""
            elif act == "attack_class":
                notice, notice_until = _attack_note(pay), time.time() + 5

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
                print("已暂停" if paused else "继续")
            elif key == ord("f"):
                notice, notice_until = _toggle_fullscreen(), time.time() + 2
            elif key == ord("p"):
                notice, notice_until = _next_profile(), time.time() + 3
            elif key == ord("h"):
                print(help_text())
            elif key == ord("r"):
                analyzer.reset("手动重置")
                embedding = None
                notice, notice_until = "轨迹已重置", time.time() + 2
            elif key == ord("s"):
                if base is None:
                    print("--no-viz 模式下没有可保存的画面。")
                else:
                    _snapshot(cfg, base)
                    notice, notice_until = "已保存截图", time.time() + 3
            elif key == ord("v"):
                notice, overlay_name, notice_until = *_cycle_overlay(cfg), time.time() + 2
            elif key == ord("m"):
                mode, sel, pending = "manage", 0, ""
                manage_notice = ""
            elif key == ord("e"):
                _gui_enroll(cfg, engine, gallery, analyzer, dash, src)
                notice, notice_until = "录入流程结束,回到识别", time.time() + 3
                embedding = None
                analyzer.reset("录入后重置")
            elif key == ord("c"):
                notice, notice_until = _switch_camera(), time.time() + 3
            elif ord("1") <= key <= ord("7"):
                notice, notice_until = _toggle_cue(CUE_KEYS[key - ord("1")]), time.time() + 2
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        metrics.close()
        journal.append(gate.events[logged_events:])
        src.release()
        cv2.destroyAllWindows()
    return 0


def _attack_note(key: str) -> str:
    from facegate.liveness.attack import ATTACK_CLASSES, SIGNATURE

    label = dict(ATTACK_CLASSES).get(key, key)
    sig = SIGNATURE.get(key, {})
    parts = []
    for k, v in sig.items():
        if v is None:
            continue
        parts.append(f"{k}{'应有活体特征' if v == 1 else '应有攻击特征'}")
    return f"「{label}」的期望签名:{'; '.join(parts)}"


def _write_metrics(metrics: MetricsWriter, frame, obs, lv, st, matches, analyzer) -> None:
    if not metrics.enabled:
        return
    head = set(MetricsWriter.HEADER)
    row: Dict[str, object] = {"t": f"{frame.stamp:.3f}", "frame": frame.index, "verdict": st.verdict}
    if lv is not None:
        row["live_score_fused"] = f"{lv.score:.4f}"
        row["live_instant"] = f"{lv.instant:.4f}"
        row["static"] = f"{lv.static:.4f}"
        row["temporal"] = "" if lv.temporal is None else f"{lv.temporal:.4f}"
        for c in lv.cues:
            if c.key in head:
                row[c.key] = "" if c.score is None else f"{c.score:.4f}"
            for k, v in c.extra.items():
                if k in head:
                    row[k] = f"{v:.5f}"
    row["similarity"] = f"{matches[0].similarity:.4f}" if matches else ""
    if analyzer.samples:
        for k in ("ear", "lap", "band_ratio", "peak_ratio", "peak_count", "skin", "sat",
                  "pitch", "yaw", "roll", "iod", "det"):
            if k in analyzer.samples[-1].m:
                row[k] = f"{analyzer.samples[-1].m[k]:.5f}"
    if obs is not None:
        row["addon_live"] = "" if obs.live_score is None else f"{obs.live_score:.4f}"
    metrics.write(row)


def _print_status(frame, obs, lv, st, matches, fps, stats) -> None:
    """--no-viz 模式下的文本状态行。"""
    sim = f"{matches[0].similarity:.3f}" if matches else "  —  "
    who = matches[0].person.name if matches else "—"
    live = f"{lv.score:.2f}" if lv else " — "
    cues = " ".join(
        f"{c.key[:4]}={'n/a' if c.score is None else format(c.score, '.2f')}"
        for c in (lv.cues if lv else [])
    )
    iod = f"{obs.iod:5.1f}" if obs is not None else "  —  "
    print(
        f"#{frame.index:5d} {fps:5.1f}fps 快{stats.fast_ms:5.1f}ms | "
        f"{st.verdict_cn:<5} sim={sim} {who[:8]:<8} 活体={live} 瞳距={iod} | {cues}"
    )


def _snapshot(cfg: Config, canvas: np.ndarray) -> None:
    d = cfg.resolve(cfg.viz.snapshot_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png"
    imwrite_u(p, canvas)
    print(f"截图已保存: {p}")


def _cycle_overlay(cfg: Config):
    v = cfg.viz
    states = [(True, True, True, True), (False, False, False, False), (True, False, False, False)]
    names = {states[0]: "全开", states[1]: "全关", states[2]: "仅人脸框"}
    cur = (v.show_landmarks, v.show_pose_axis, v.show_homography, v.show_eye_axis)
    try:
        nxt = states[(states.index(cur) + 1) % len(states)]
    except ValueError:
        nxt = states[0]
    v.show_landmarks, v.show_pose_axis, v.show_homography, v.show_eye_axis = nxt
    name = names[nxt]
    print(f"叠加层预设: {name}")
    return f"叠加层:{name}", name


# ===========================================================================
# 录入
# ===========================================================================


def _capture_samples(
    cfg: Config,
    engine: FaceEngine,
    gallery: Gallery,
    analyzer: LivenessAnalyzer,
    dash: Dashboard,
    src,
    key: str,
    name: str,
) -> int:
    """采集样本直到达到 samples_per_person 条,返回成功条数。

    两次采集之间强制间隔 ``CAPTURE_COOLDOWN`` 秒:否则条件一满足就会每帧都调一次
    ArcFace(约 100ms),既拖慢画面,又会因为姿态几乎不变而全部被相似度去重挡掉。
    """
    CAPTURE_COOLDOWN = 0.8
    want = cfg.gallery.samples_per_person
    got = 0
    pose_hint = 0
    hints = [
        "请正对摄像头",
        "请把头转向左侧约 20 度",
        "请把头转向右侧约 20 度",
        "请抬头约 15 度",
        "请低头约 15 度",
        "请保持自然表情",
    ]
    fps = FpsMeter()
    gate = Gate(cfg, gallery)
    last_try = 0.0
    analyzer.reset("开始录入")
    _open_window()
    print(f"\n开始录入 [{name}] (id={key}),目标 {want} 条样本。按 q 放弃。")

    while got < want:
        frame = src.read()
        if frame is None:
            print("视频源结束或摄像头断开。")
            break
        image = frame.image

        obs_list = engine.detect(image)
        obs = obs_list[0] if obs_list else None
        if obs is None:
            analyzer.mark_lost()
        else:
            analyzer.push(obs, image, frame.index)
        lv = analyzer.evaluate() if analyzer.samples else None
        st = gate.update(obs is not None, lv, None, 0.0)

        reason = _enroll_reject(cfg, obs, lv)
        now = time.perf_counter()
        if reason is None and obs is not None and (now - last_try) >= CAPTURE_COOLDOWN:
            last_try = now
            ok = _try_add(cfg, engine, gallery, obs, lv, image, key, name, got)
            if ok:
                got += 1
                pose_hint += 1
            else:
                reason = "姿态与已有样本过于接近,请按提示换个角度"
        if reason:
            if frame.index % 30 == 0:
                print(f"  等待中: {reason}")

        hint = hints[min(pose_hint, len(hints) - 1)]
        canvas = dash.render(
            image, obs, st, lv, analyzer, [], fps.tick(), fps.value, engine.stats, src.label,
            similarity=0.0, gallery_total=len(gallery),
            notice=(f"采集 {got}/{want} · {hint}" if not reason else f"采集 {got}/{want} · {reason}"),
        )
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(1) & 0xFF
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("窗口已关闭,放弃录入。")
                break
        except cv2.error:
            break
        if k in (ord("q"), 27):
            print("已放弃录入。")
            break

    if got:
        print(f"\n录入完成:[{name}] id={key} 共 {got} 条样本 → {cfg.gallery_dir}")
    return got


def _enroll_reject(cfg: Config, obs: Optional[FaceObs], lv) -> Optional[str]:
    if obs is None:
        return "没有检测到人脸"
    g = cfg.gallery
    w, h = obs.bbox[2] - obs.bbox[0], obs.bbox[3] - obs.bbox[1]
    if min(w, h) < g.min_face_px:
        return f"人脸太小({int(min(w, h))}px < {g.min_face_px}px),请靠近摄像头"
    if obs.det_score < g.min_det_score:
        return f"检测置信度不足({obs.det_score:.2f} < {g.min_det_score:.2f})"
    if lv is None or lv.verdict == "observing":
        return "活体证据积累中,请缓慢左右转头"
    if g.require_live and lv.verdict != "live":
        return f"活体检测未通过(融合分 {lv.score:.2f}),不允许用照片或屏幕录入"
    return None


def _try_add(cfg, engine, gallery, obs, lv, image, key, name, got) -> bool:
    try:
        emb = engine.embed_crop(obs.aligned)
    except Exception as exc:  # noqa: BLE001
        print(f"  特征提取失败: {exc}")
        return False
    extra = {
        "live_score": float(lv.score) if lv else 0.0,
        "pitch": float(obs.pose[0]) if obs.pose is not None else 0.0,
        "yaw": float(obs.pose[1]) if obs.pose is not None else 0.0,
        "roll": float(obs.pose[2]) if obs.pose is not None else 0.0,
        "laplacian": 0.0,
    }
    thumb = _thumb(image, obs) if got == 0 else None
    try:
        gallery.add_sample(key, name, emb, obs.det_score, thumb, extra)
    except DuplicateSample:
        return False
    print(
        f"  [{got + 1}] 已保存  yaw={extra['yaw']:+6.1f}°  pitch={extra['pitch']:+6.1f}°  "
        f"det={obs.det_score:.2f}  活体={extra['live_score']:.2f}"
    )
    return True


def _thumb(image: np.ndarray, obs: FaceObs, size: int = 160) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in obs.bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _gui_enroll(cfg, engine, gallery, analyzer, dash, src) -> None:
    """在 GUI 内录入:弹框问姓名,然后进入采集循环。"""
    print("\n" + "=" * 60)
    cv2.destroyWindow(WINDOW)
    raw = _ask_name("录入人脸")
    if not raw:
        print("已取消录入。")
        _open_window()
        return
    _capture_samples(cfg, engine, gallery, analyzer, dash, src, sanitize_id(raw), raw)
    _open_window()


def cmd_enroll(cfg: Config, args) -> int:
    if not args.name:
        try:
            args.name = input("请输入姓名或 ID: ").strip()
        except EOFError:
            args.name = ""
    if not args.name:
        print("必须提供 --name。")
        return 2
    key = sanitize_id(args.id or args.name)
    engine, gallery, analyzer, dash, src = build(cfg)
    try:
        n = _capture_samples(cfg, engine, gallery, analyzer, dash, src, key, args.name)
    finally:
        src.release()
        cv2.destroyAllWindows()
    print_table(cfg, gallery)
    return 0 if n else 1


# ===========================================================================
# 管理
# ===========================================================================


def print_table(cfg: Config, gallery: Gallery) -> None:
    from prettytable import PrettyTable

    t = PrettyTable()
    t.field_names = ["ID", "姓名", "样本数", "平均检测分", "创建时间"]
    t.align = "l"
    for p in sorted(gallery.people.values(), key=lambda x: x.created_at):
        t.add_row([p.id, p.name, p.samples, f"{p.mean_det_score:.3f}", time.strftime("%Y-%m-%d %H:%M", time.localtime(p.created_at))])
    print(f"\n人脸库位置: {cfg.gallery_dir}")
    print(t if len(gallery) else "(空)")
    print(f"合计 {len(gallery)} 人 / {gallery.total_samples} 条特征向量\n")


def cmd_list(cfg: Config) -> int:
    gallery = Gallery.open(cfg.gallery, cfg.gallery_dir)
    print_table(cfg, gallery)
    return 0


def cmd_remove(cfg: Config, args) -> int:
    gallery = Gallery.open(cfg.gallery, cfg.gallery_dir)
    key = args.name or args.id
    if not key:
        print("需要 --name 或 --id。")
        return 2
    targets = gallery.find(key)
    if not targets:
        print(f"没有找到匹配 “{key}” 的记录。")
        print_table(cfg, gallery)
        return 1
    names = ", ".join(f"{p.name}(id={p.id})" for p in targets)
    if not args.yes:
        try:
            ok = input(f"确认删除 {names} ? [y/N] ").strip().lower()
        except EOFError:
            ok = ""
        if ok not in ("y", "yes"):
            print("已取消。")
            return 0
    removed = gallery.remove(key)
    print(f"已删除 {len(removed)} 人: {', '.join(p.id for p in removed)}")
    print_table(cfg, gallery)
    return 0


def cmd_clear(cfg: Config, args) -> int:
    gallery = Gallery.open(cfg.gallery, cfg.gallery_dir)
    if not len(gallery):
        print("人脸库已经是空的。")
        return 0
    if not args.yes:
        bak = gallery.backup()
        print(f"已备份到 {bak}")
        try:
            ok = input(f"确认清空 {len(gallery)} 人的整个人脸库? [y/N] ").strip().lower()
        except EOFError:
            ok = ""
        if ok not in ("y", "yes"):
            print("已取消。")
            return 0
    n = gallery.clear()
    print(f"已清空人脸库({n} 人)。位置: {cfg.gallery_dir}")
    return 0


# ===========================================================================
# 入口
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="单目 RGB 摄像头人脸门禁 Demo(insightface + 多线索活体检测)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-c", "--config", default=None, help="配置文件路径,默认 ./config.yaml")
    p.add_argument("--camera", type=int, default=None, help="覆盖 camera.index")
    p.add_argument("--source", default=None, help="覆盖 camera.source,用视频文件做攻击测试")
    p.add_argument("--det-size", type=int, default=None, help="覆盖检测输入边长,如 320")
    p.add_argument("--threads", type=int, default=None, help="覆盖 ONNX Runtime intra_op 线程数")
    p.add_argument("--perf", choices=["quality", "balanced", "fast", "ultra"], default=None,
                   help="性能档:quality / balanced / fast / ultra(精简面板,帧率最高);"
                        "运行中也可按 p 或点工具栏切换")
    p.add_argument("--no-viz", action="store_true", help="不显示窗口,只在终端输出")

    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("recognize", help="实时门禁识别")
    sp.set_defaults(func=lambda cfg, a: cmd_recognize(cfg))

    sp = sub.add_parser("enroll", help="录入人脸")
    sp.add_argument("--name", default=None, help="姓名(可中文)")
    sp.add_argument("--id", default=None, help="自定义 ID,默认由姓名生成")
    sp.set_defaults(func=cmd_enroll)

    sp = sub.add_parser("list", help="查看人脸库")
    sp.set_defaults(func=lambda cfg, a: cmd_list(cfg))

    sp = sub.add_parser("remove", help="按 ID / 姓名删除")
    sp.add_argument("--name", default=None)
    sp.add_argument("--id", default=None)
    sp.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    sp.set_defaults(func=cmd_remove)

    sp = sub.add_parser("clear", help="清空整个人脸库")
    sp.add_argument("-y", "--yes", action="store_true", help="跳过确认与备份")
    sp.set_defaults(func=cmd_clear)

    sp = sub.add_parser("selftest", help="环境自检与阈值标定")
    sp.add_argument("--dump-config", action="store_true", help="同时打印完整参数快照")
    sp.set_defaults(func=lambda cfg, a: _selftest(cfg, a))

    sp = sub.add_parser("cameras", help="列出可用摄像头序号与分辨率")
    sp.set_defaults(func=lambda cfg, a: cmd_cameras())

    return p


def cmd_cameras() -> int:
    cams = list_cameras()
    if not cams:
        print("没有探测到可用摄像头。")
        return 1
    print("可用摄像头:")
    for i, (w, h) in cams:
        print(f"  [{i}]  {w}×{h}    启动:python main.py --camera {i} recognize")
    print("运行中也可按 c 或点工具栏「摄像头:n」切换。")
    return 0


def _selftest(cfg: Config, args) -> int:
    from facegate.selftest import run_all

    rc = run_all(cfg)
    if args.dump_config:
        print("\n" + "=" * 76 + "\n当前生效的全部参数\n" + "=" * 76)
        print(dump_config(cfg))
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, ROOT)

    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.source is not None:
        cfg.camera.source = args.source
    if args.det_size is not None:
        cfg.engine.det_size = [args.det_size, args.det_size]
    if args.threads is not None:
        cfg.engine.intra_op_threads = args.threads
    if args.perf is not None:
        cfg.engine.perf_profile = args.perf
    cfg.viz.minimal = bool(PROFILE_MINIMAL_VIZ.get(cfg.engine.perf_profile, False))
    if args.no_viz:
        cfg.viz.enabled = False

    cfg.gallery_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"项目根目录: {ROOT}")
    print(f"配置文件:   {ROOT / 'config.yaml'}{'(未找到,使用内置默认值)' if not (ROOT / 'config.yaml').exists() else ''}")
    print(f"人脸库:     {cfg.gallery_dir}")

    if args.cmd is None:
        return cmd_recognize(cfg)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
