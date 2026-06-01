from __future__ import annotations

import argparse
from pathlib import Path

import cv2

SUPPORTED_SUFFIXES = {".mov"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 .mov 视频抽帧保存为图片。支持单个视频文件或目录递归处理。")
    parser.add_argument(
        "input_path",
        type=Path,
        help="输入的 .mov 文件，或包含多个 .mov 文件的目录",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="输出图片目录。若输入是目录，会在该目录下为每个视频创建子目录。",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        help="每隔多少帧保存一次，默认每一帧都保存。",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        choices=["jpg", "png"],
        help="输出图片格式，默认 jpg。",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG 质量（1-100），仅在 --image-ext=jpg 时生效。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多导出多少张图片，默认不限制。",
    )
    return parser.parse_args()


def find_mov_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"只支持 {SUPPORTED_SUFFIXES} 文件，当前输入为: {input_path}")
        return [input_path]

    return sorted(
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def export_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int,
    image_ext: str,
    jpg_quality: int,
    max_frames: int | None,
) -> int:
    if every_n_frames <= 0:
        raise ValueError("--every-n-frames 必须大于 0")
    if image_ext == "jpg" and not 1 <= jpg_quality <= 100:
        raise ValueError("--jpg-quality 必须在 1 到 100 之间")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_index = 0
    saved_count = 0

    if image_ext == "jpg":
        write_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    else:
        write_params = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % every_n_frames == 0:
            image_path = output_dir / f"frame_{saved_count:06d}.{image_ext}"
            write_ok = cv2.imwrite(str(image_path), frame, write_params)
            if not write_ok:
                capture.release()
                raise RuntimeError(f"保存图片失败: {image_path}")

            saved_count += 1
            if max_frames is not None and saved_count >= max_frames:
                break

        frame_index += 1

    capture.release()
    return saved_count


def build_output_dir(video_path: Path, input_path: Path, output_root: Path) -> Path:
    if input_path.is_file():
        return output_root

    relative_parent = video_path.parent.relative_to(input_path)
    return output_root / relative_parent / video_path.stem


def main() -> None:
    args = parse_args()
    mov_files = find_mov_files(args.input_path)

    if not mov_files:
        raise FileNotFoundError(f"没有找到 .mov 文件: {args.input_path}")

    total_saved = 0
    for video_path in mov_files:
        target_dir = build_output_dir(video_path, args.input_path, args.output_dir)
        saved = export_frames(
            video_path=video_path,
            output_dir=target_dir,
            every_n_frames=args.every_n_frames,
            image_ext=args.image_ext,
            jpg_quality=args.jpg_quality,
            max_frames=args.max_frames,
        )
        total_saved += saved
        print(f"[完成] {video_path} -> {target_dir}，导出 {saved} 张图片")

    print(f"全部处理完成，共导出 {total_saved} 张图片。")


if __name__ == "__main__":
    main()
