import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract character dictionary from a PP-OCR YAML configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the PP-OCR YAML configuration file (e.g. inference.yml)",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to save the character dictionary text file",
    )
    return parser.parse_args()


def extract_character_dict(config_path: Path, output_path: Path) -> list[str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"YAML configuration file not found: {config_path}")

    print(f"Reading configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chars = data.get("PostProcess", {}).get("character_dict")
    if chars is None:
        chars = data.get("character_dict")

    if not chars:
        raise ValueError(f"Could not find 'character_dict' in {config_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(f"{char}\n" for char in chars)

    print(f"Successfully extracted {len(chars)} characters to: {output_path}")
    return chars


def main() -> None:
    args = parse_args()
    extract_character_dict(args.config_path, args.output_path)


if __name__ == "__main__":
    main()
