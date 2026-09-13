import argparse
from pathlib import Path

import onnx
from onnx import version_converter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade an ONNX model to a target opset version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_model",
        type=Path,
        help="Path to the input ONNX model file",
    )
    parser.add_argument(
        "output_model",
        type=Path,
        help="Path to save the upgraded ONNX model file",
    )
    parser.add_argument(
        "--target-opset",
        type=int,
        default=17,
        help="Target ONNX opset version",
    )
    return parser.parse_args()


def upgrade_opset(input_path: Path, output_path: Path, target_opset: int = 17) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model from: {input_path}")
    model = onnx.load(str(input_path))

    opsets = {op.domain: op.version for op in model.opset_import}
    current_opset = opsets.get("", opsets.get("ai.onnx"))
    print(f"Current opset version: {current_opset}")

    if current_opset == target_opset:
        print(f"Model is already opset {target_opset}. Saving to: {output_path}")
        onnx.save(model, str(output_path))
    else:
        print(f"Upgrading opset version from {current_opset} to {target_opset}...")
        converted_model = version_converter.convert_version(model, target_opset)
        onnx.checker.check_model(converted_model)
        onnx.save(converted_model, str(output_path))

    print(f"Successfully saved upgraded model to: {output_path}")


def main() -> None:
    args = parse_args()
    upgrade_opset(args.input_model, args.output_model, args.target_opset)


if __name__ == "__main__":
    main()
