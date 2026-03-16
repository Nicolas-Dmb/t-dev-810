import sys

from t_dev_810.experiments import build_config


def main():
    try:
        experimentConf = build_config()
    except Exception as e:
        print(e)
        sys.exit

    print(f" experiment result : {experimentConf}")


if __name__ == "__main__":
    main()
