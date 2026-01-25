import argparse
from src.runner import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
