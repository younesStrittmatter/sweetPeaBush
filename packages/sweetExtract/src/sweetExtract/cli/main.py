import argparse
from pathlib import Path
from sweetExtract.pdf_reader import extract_text
from sweetExtract.pipeline.list_experiments import run as list_exps
from sweetExtract.pipeline.describe_experiment import run as describe
from sweetExtract.io_utils.save import save_json, save_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to paper PDF")
    ap.add_argument("--outdir", default="out", help="Output folder")
    args = ap.parse_args()

    text = extract_text(args.pdf)
    stubs = list_exps(text)
    if not stubs:
        print("No experiments found.")
        return

    full = [describe(s, text) for s in stubs]

    outdir = Path(args.outdir)
    save_json(stubs, outdir / "experiments_list.json")
    save_json(full, outdir / "experiments_full.json")
    save_markdown(full, outdir / (Path(args.pdf).stem + "_experiments.md"))
    print(f"Saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
