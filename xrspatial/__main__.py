def main(args=None):
    try:
        import pyct.cmd
    except ImportError:
        import sys

        from . import _missing_cmd
        print(_missing_cmd())
        sys.exit(1)
    try:
        return pyct.cmd.substitute_main('xrspatial', args=args)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
