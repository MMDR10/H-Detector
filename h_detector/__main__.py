"""
CLI entry point for H-Detector.

Usage:
    python -m h_detector --help
    python -m h_detector --model Qwen2.5-1.5B --prompt "Your text here"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="H-Detector: Geometric detection of LLM internal anomalies",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen2.5-1.5B",
        help="Model name to load (default: Qwen2.5-1.5B)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Input prompt to analyze"
    )
    parser.add_argument(
        "--prompt-file", type=str, default=None,
        help="File containing input prompt"
    )
    parser.add_argument(
        "--layer-h", action="store_true",
        help="Output per-layer H values"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (cpu or cuda)"
    )

    args = parser.parse_args()

    # Get prompt text
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"❌ File not found: {args.prompt_file}")
            sys.exit(1)
    elif args.prompt:
        prompt = args.prompt
    else:
        # Interactive mode
        print("H-Detector Interactive Mode 🛡️")
        print("Enter a prompt to analyze (Ctrl+D to exit):")
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("No input provided.")
            sys.exit(0)

    print(f"\n🔬 Analyzing: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"📐 Model: {args.model}")
    print(f"\n⚠️  To run full detection, you need access to the model's hidden states.")
    print(f"   See https://github.com/MMDR10/H-Detector for integration guide.\n")


if __name__ == "__main__":
    main()
