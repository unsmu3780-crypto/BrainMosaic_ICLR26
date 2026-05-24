import sys

from build_zuco_text_assets_inputs import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--task", "ZuCoTSR"]
    main()
