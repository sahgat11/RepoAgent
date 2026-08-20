import sys

from repoagent.loader import load_repository


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <repo-path>")
        return

    repo_path = sys.argv[1]

    files = load_repository(repo_path)

    print(f"Found {len(files)} source files.\n")

    for file in files:
        print(file)


if __name__ == "__main__":
    main()