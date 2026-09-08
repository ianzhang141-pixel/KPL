"""命令行入口：python3 -m kplab <子命令>

只依赖 Mac 自带的 python3，不需要 pip 安装任何东西。
所有子命令都可以加 --data 指定项目的 data 目录，不加就自动往上找。

    python3 -m kplab serve                      打开控制台（推荐，其余全在网页上点）
    python3 -m kplab where                      看看它认的是哪个 data 目录
    python3 -m kplab doctor                     自检：ffmpeg / 识别后端 / 待核对常量
    python3 -m kplab new  <比赛编号>             建一场新比赛
    python3 -m kplab frames <比赛编号> --video X 从录像里抽帧
    python3 -m kplab check <比赛编号>            数据完整性与可信度检查
    python3 -m kplab state <比赛编号>            观测 → 分钟级 State
    python3 -m kplab evaluate <比赛编号>         人工标注 vs 机器识别，看识别准不准
    python3 -m kplab rules                      查看 / 核对游戏常量
    python3 -m kplab minimap <帧图>              小地图找人（认位置，不认英雄）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (__version__, annotate, check, evaluate, hud, minimap, ocr, paths,
               rules, schema, state as state_mod, store, video)


def cmd_where(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    print(f"data 目录：{data_dir}")
    print(f"存在：{'是' if data_dir.is_dir() else '否（还没创建）'}")
    if not data_dir.is_dir():
        print("如果这不是你想用的目录，请用 --data 指定，例如：")
        print("  python3 -m kplab where --data ~/王者数据/data")
        return 0
    inv = store.inventory(data_dir)
    print(f"比赛 {inv['total']} 场    有观测 {inv['withObservations']} 场    "
          f"有 State {inv['withState']} 场    占用 {inv['totalBytesHuman']}")
    for game in inv["games"][:20]:
        flags = []
        if game["observationCount"]:
            flags.append(f"观测{game['observationCount']}")
        if game["annotationCount"]:
            flags.append(f"标注{game['annotationCount']}")
        if game["frameImages"]:
            flags.append(f"帧图{game['frameImages']}")
        print(f"  · {game['gameId']:<32}{'  '.join(flags) or '（空）'}")
    if inv["total"] > 20:
        print(f"  …… 还有 {inv['total'] - 20} 场")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    print(f"kplab {__version__}")
    print(f"data 目录：{data_dir}（{'存在' if data_dir.is_dir() else '不存在'}）")
    print()

    info = video.available()
    print(f"抽帧（ffmpeg）：{'✅ ' + (info['ffmpeg'] or '') if info['ok'] else '❌ 未安装'}")
    if not info["ok"]:
        for line in info["hint"].splitlines():
            print(f"    {line}")

    print()
    print("自动识别后端：")
    for backend in ocr.backends():
        mark = "✅" if backend["available"] else "❌"
        print(f"  {mark} {backend['label']:<12}{backend['note']}")
        if not backend["available"] and backend["install"]:
            print(f"       安装：{backend['install']}")

    print()
    print(ocr.status()["message"])

    print()
    table = rules.load(data_dir)
    pending = rules.unverified(table)
    if pending:
        print(f"⚠️  有 {len(pending)} / {len(table)} 条游戏常量还没核对：")
        for key in pending:
            entry = table[key]
            print(f"    · {key:<26}{entry['what']}（当前值 {entry['value']}）")
        print("    核对办法：python3 -m kplab rules --set <名字>=<值>")
    else:
        print("✅ 游戏常量已全部核对。")

    print()
    profiles = hud.load_profiles(data_dir)
    print("HUD 标定档案：")
    for name, profile in profiles.items():
        verdict = hud.describe(profile)
        print(f"  {'✅' if verdict['ok'] else '❌'} {name:<18}{verdict['message']}")

    print()
    print(ocr.minimap_note())
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    data_dir = paths.ensure(paths.find_data_dir(args.data))
    if not store.valid_game_id(args.game_id):
        print(f"❌ 比赛编号不合法：{args.game_id}")
        print("   只能用字母、数字、下划线、点、短横线。建议：KPL2026S1_AG_vs_EST_G1")
        return 2
    meta = {"title": args.title or "", "tournament": args.tournament or "",
            "blueTeam": args.blue or "", "redTeam": args.red or "",
            "sourceKind": args.source or "", "sourceUrl": args.url or ""}
    if args.blue_win is not None:
        meta["blueWin"] = args.blue_win
    path = store.create_game(data_dir, args.game_id, meta)
    print(f"✅ 已建立：{path.parent}")
    if args.blue_win is None:
        print("⚠️  还没记录胜负。胜负是将来训练模型的唯一标签，建议现在就补：")
        print(f"     python3 -m kplab new {args.game_id} --blue-win / --red-win")
    print()
    print("下一步：")
    print(f"  有录像 → python3 -m kplab frames {args.game_id} --video ~/下载/xxx.mp4")
    print("  没录像 → python3 -m kplab serve，在标注页上边看录像边填")
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    data_dir = paths.ensure(paths.find_data_dir(args.data))
    if not store.load_meta(data_dir, args.game_id):
        print(f"❌ 没有这场比赛：{args.game_id}")
        print(f"   先建立：python3 -m kplab new {args.game_id}")
        return 2
    source = Path(args.video).expanduser()
    try:
        info = video.probe(source)
        print(f"录像：{source.name}")
        print(f"  {info.width}×{info.height}    {info.durationSec / 60:.1f} 分钟    {info.fps:.1f} fps")
        out_dir = store.frames_dir(data_dir, args.game_id)
        print(f"  每 {args.every} 秒抽一帧 → {out_dir}")

        def progress(done: int, total: int) -> None:
            print(f"\r  {done}/{total}", end="", flush=True)

        records = video.extract_frames(
            source, out_dir, args.every, args.start, args.end, on_progress=progress
        )
    except video.VideoError as err:
        print(f"❌ {err}")
        return 2
    good = [r for r in records if r.get("file")]
    print(f"\n✅ 抽出 {len(good)} 帧，失败 {len(records) - len(good)} 帧")
    store.create_game(data_dir, args.game_id, {
        "videoPath": str(source), "videoWidth": info.width,
        "videoHeight": info.height, "frameEverySec": args.every,
    })
    print()
    print("注意：抽帧只是把画面切出来了，里面还没有任何数字。")
    print("下一步：python3 -m kplab serve，在标注页上逐帧填数。")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    report = check.run(data_dir, args.game_id)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(check.format_text(report))
    return 0 if report["verdict"]["ok"] else 1


def cmd_state(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    meta = store.load_meta(data_dir, args.game_id)
    if not meta:
        print(f"❌ 没有这场比赛：{args.game_id}")
        return 2
    observations = list(store.read_jsonl_gz(store.observations_path(data_dir, args.game_id)))
    if not observations:
        print("❌ 这场还没有任何观测。先去标注：python3 -m kplab serve")
        return 2

    built = state_mod.build(meta, observations, rules.load(data_dir))
    out = Path(args.out).expanduser() if args.out else store.state_path(data_dir, args.game_id)
    store.write_json_gz(out, built)
    cov = built["coverage"]
    print(f"✅ 已生成 {built['meta']['frameCount']} 帧 State：{out}")
    print(f"   字段覆盖 {cov['knownRatio'] * 100:.0f}%、可信 {cov['trustedRatio'] * 100:.0f}%、"
          f"可用帧 {cov['usableFrames']}/{cov['frames']}")
    if built["rejections"]:
        print(f"   {len(built['rejections'])} 个读数因为与已知的过去矛盾被拒绝（这是校验在干活）")
    if args.minute is not None:
        _print_frame(built, args.minute)
    return 0


def _print_frame(built: dict, minute: float) -> None:
    frames = built["frames"]
    if not frames:
        return
    frame = min(frames, key=lambda f: abs(f["minute"] - minute))
    print()
    print("=" * 78)
    print(f"{frame['clock']}   （第 {frame['frameIndex']} 帧）"
          f"    字段覆盖 {frame['coverage']['knownRatio'] * 100:.0f}%")
    print("=" * 78)

    def show(entry: dict) -> str:
        """显示一个字段。**存疑的必须标出来**，不能和事实长得一样。"""
        if not schema.is_known(entry):
            return "—"
        value = schema.get(entry)
        return str(value) if schema.trusted(entry) else f"{value}?"

    for team_id, label in ((str(rules.BLUE), "蓝方"), (str(rules.RED), "红方")):
        team = frame["teams"][team_id]
        print(f"{label}  经济 {show(team['totalGold']):>7}  击杀 {show(team['kills']):>3}  "
              f"塔 {show(team['towers'])}  暴君 {show(team['tyrants'])}  "
              f"黑暗暴君 {show(team['darkTyrants'])}  主宰 {show(team['overlords'])}  "
              f"风暴龙王 {show(team['stormDragons'])}")
    print(f"经济差（蓝-红）：{show(frame['diff']['totalGold'])}")
    print("-" * 78)
    print(f"{'':<3}{'英雄':<12}{'分路':<8}{'Lv':>4}{'经济':>8}{'K':>4}{'D':>4}{'A':>4}   小地图")
    for player in frame["players"]:
        side = "蓝" if player["teamId"] == rules.BLUE else "红"
        x, y = schema.get(player["x"]), schema.get(player["y"])
        position = f"({x:.2f}, {y:.2f})" if x is not None and y is not None else "—"
        print(f"{side:<3}{(player['heroName'] or '?'):<12}{(player['role'] or '?'):<8}"
              f"{show(player['level']):>4}{show(player['totalGold']):>8}"
              f"{show(player['kills']):>4}{show(player['deaths']):>4}"
              f"{show(player['assists']):>4}   {position}")
    print("-" * 78)
    print("带 ? 的是置信度不够的值，只能参考，不能当事实用。「—」表示不知道。")
    if frame["recentEvents"]:
        print(f"这一段里的事件（{len(frame['recentEvents'])} 条）：")
        for event in frame["recentEvents"][:20]:
            print(f"  {event['clock']}  {event['type']}  "
                  f"{'蓝' if event.get('teamId') == rules.BLUE else '红'}方")


def cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    report = evaluate.run(data_dir, args.game_id)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(evaluate.format_text(report))
    return 0 if report["verdict"]["ok"] else 1


def cmd_rules(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    if args.set:
        if "=" not in args.set:
            print("❌ 格式是 --set 名字=值，例如 --set towersPerLane=2")
            return 2
        key, _, raw = args.set.partition("=")
        key = key.strip()
        if key not in rules.DEFAULTS:
            print(f"❌ 没有这条常量：{key}")
            print(f"   可选：{'、'.join(rules.DEFAULTS)}")
            return 2
        try:
            value = int(raw.strip())
        except ValueError:
            print(f"❌ 值必须是整数：{raw!r}")
            return 2
        path = rules.save_override(paths.ensure(data_dir), key, value, args.by or "命令行核对")
        print(f"✅ 已记录 {key} = {value}（已标记为核对过）：{path}")
        return 0

    table = rules.load(data_dir)
    print(f"{'名字':<26}{'当前值':>8}  {'核对':<6}说明")
    print("-" * 92)
    for key, entry in table.items():
        mark = "✅" if entry.get("verified") else "待核对"
        print(f"{key:<26}{str(entry['value']):>8}  {mark:<6}{entry['what']}")
    pending = rules.unverified(table)
    print("-" * 92)
    if pending:
        print(f"还有 {len(pending)} 条没核对。这些是**凭印象写的默认值**，不是查证过的事实；")
        print("版本更新后很可能不对，而错误的常量会让 check 的上限校验跟着失效。")
        print("核对后用：python3 -m kplab rules --set towersPerLane=2")
    else:
        print("全部已核对。")
    return 0


def cmd_minimap(args: argparse.Namespace) -> int:
    data_dir = paths.find_data_dir(args.data)
    image = Path(args.image).expanduser()
    thresholds = minimap.load_thresholds(data_dir)

    box = None
    if args.box:
        try:
            parts = [float(v) for v in args.box.split(",")]
        except ValueError:
            print("❌ --box 要写成四个数字，例如 0,0.66,0.185,0.33")
            return 2
        if len(parts) != 4:
            print("❌ --box 需要正好四个数字：x,y,宽,高（都是 0~1 的相对值）")
            return 2
        box = (parts[0], parts[1], parts[2], parts[3])
    else:
        profile = hud.get_profile(data_dir, args.profile)
        region = (profile or {}).get("regions", {}).get("minimap")
        if not region:
            print(f"❌ 标定档案「{args.profile}」里没有小地图区域，也没给 --box。")
            print("   用 --box 手动指定，例如：--box 0,0.66,0.185,0.33")
            return 2
        box = tuple(region)

    try:
        result = minimap.detect(image, box, thresholds)
    except Exception as err:      # noqa: BLE001 - 要让用户看见原因
        print(f"❌ {type(err).__name__}: {err}")
        return 2

    print(f"图片：{image}")
    print(f"小地图区域：{box}")
    print()
    print(minimap.summary(result))

    if args.calibrate:
        if not result["quality"]["usable"]:
            print()
            print("❌ 这次识别本身就不可用，不能拿它当标定结果。")
            print("   先调 --box 或阈值，等识别结果合理了再 --calibrate。")
            return 1
        path = minimap.save_thresholds(data_dir, thresholds, verified=True)
        print()
        print(f"✅ 已把当前阈值标记为「已标定」：{path}")
        print("   注意：这只表示你确认过这一帧的识别结果合理，")
        print("   换一个录像源（画质、色调不同）可能要重新标定。")
    return 0 if result["quality"]["usable"] else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server
    server.serve(paths.find_data_dir(args.data), port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m kplab",
        description="王者荣耀决策实验室 —— 数据验证与状态抽取工具箱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", help="项目的 data 目录（不写就自动往上找）")
    parser.add_argument("--version", action="version", version=f"kplab {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("where", help="显示它认的是哪个 data 目录，以及有哪些比赛")
    p.set_defaults(func=cmd_where)

    p = sub.add_parser("doctor", help="自检：ffmpeg、识别后端、待核对常量、HUD 标定")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("new", help="建立一场新比赛")
    p.add_argument("game_id", help="例如 KPL2026S1_AG_vs_EST_G1")
    p.add_argument("--title"), p.add_argument("--tournament", help="赛事名")
    p.add_argument("--blue", help="蓝方战队"), p.add_argument("--red", help="红方战队")
    p.add_argument("--source", help="数据来源：broadcast（转播）/ replay（回放）/ manual")
    p.add_argument("--url", help="录像地址，只作记录")
    p.add_argument("--blue-win", dest="blue_win", action="store_true", default=None, help="蓝方获胜")
    p.add_argument("--red-win", dest="blue_win", action="store_false", help="红方获胜")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("frames", help="从录像里按固定间隔抽帧（需要 ffmpeg）")
    p.add_argument("game_id"), p.add_argument("--video", required=True, help="录像文件路径")
    p.add_argument("--every", type=float, default=30.0, help="每几秒抽一帧，默认 30")
    p.add_argument("--start", type=float, default=0.0, help="从第几秒开始")
    p.add_argument("--end", type=float, default=None, help="到第几秒结束")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("check", help="数据完整性与可信度检查")
    p.add_argument("game_id"), p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("state", help="观测 → 分钟级 State")
    p.add_argument("game_id")
    p.add_argument("--minute", type=float, help="顺便打印这一分钟的完整局势")
    p.add_argument("--out", help="输出路径（默认 data/kpl/games/<编号>/state.json.gz）")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("evaluate", help="人工标注 vs 机器识别，看识别准不准")
    p.add_argument("game_id"), p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("rules", help="查看 / 核对游戏常量")
    p.add_argument("--set", help="记录一条核对结果，例如 towersPerLane=2")
    p.add_argument("--by", help="谁核对的")
    p.set_defaults(func=cmd_rules)

    p = sub.add_parser("minimap", help="小地图找人：认出有几个蓝/红标记、各在哪")
    p.add_argument("image", help="一张画面（抽出来的帧图）")
    p.add_argument("--box", help="小地图区域 x,y,宽,高（0~1 相对值）。不写就用标定档案里的")
    p.add_argument("--profile", default="kpl_broadcast", help="用哪个 HUD 标定档案")
    p.add_argument("--calibrate", action="store_true",
                   help="识别结果合理的话，把当前阈值标记为已标定")
    p.set_defaults(func=cmd_minimap)

    p = sub.add_parser("serve", help="打开网页控制台（推荐）")
    p.add_argument("--port", type=int, default=8020,
                   help="默认 8020，不占用 LOL 那边的 8000 / 8010")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已取消。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
