"""CLI entry point for sleep-detector-sdk."""

import argparse
import sys
from typing import List, Optional

from sleep_detector_sdk.model_manager import ModelManager


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error).
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="sleep-detector-sdk",
        description="Sleep Detector SDK — real-time drowsiness detection",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download-model command
    dl_parser = subparsers.add_parser(
        "download-model",
        help="Download the dlib facial landmark model",
    )
    dl_parser.add_argument(
        "--path",
        default=None,
        help="Directory to store the model (default: ~/.sleep-detector-sdk/models/)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "download-model":
        return _download_model(args.path)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


def _download_model(path: Optional[str]) -> int:
    """Handle the download-model command."""
    mm = ModelManager(cache_dir=path)

    if mm.is_cached:
        print(f"Model already downloaded at: {mm.model_path}")
        return 0

    print("Downloading dlib facial landmark model...")

    def progress(downloaded, total):
        if total > 0:
            pct = (downloaded / total) * 100
            print(f"\r  Progress: {pct:.1f}% ({downloaded}/{total} bytes)", end="", flush=True)

    model_path = mm.download(progress_callback=progress)
    print(f"\nModel saved to: {model_path}")
    return 0
