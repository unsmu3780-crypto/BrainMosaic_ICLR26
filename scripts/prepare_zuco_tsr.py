import sys

from prepare_zuco_task import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--task", "ZuCoTSR"]
    main()
