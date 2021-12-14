import argparse
import re


def setup(
    message: str,
    python_version: str,
    build: str,
    test: str,
):
    build_doc = "true"

    python_version = set(python_version.split(","))
    build = set(build.split(","))
    test = set(test.split(","))

    oses = {"macos", "ubuntu", "windows"}

    if "${{ github.event_name }}" != "schedule":
        options_search = re.search("\[ci:(.*)\]", message, re.IGNORECASE)

        options = []
        if options_search:
            options = [s.strip().lower() for s in options_search.group(1).split(",")]

        python_options = (
            python_version
            | {x.split("-")[1] for x in options if x.startswith("+python")}
        ) - {x.split("-")[2] for x in options if x.startswith("-python")}

        test_final = (
            test
            | {
                x[1:]
                for x in options
                if x.startswith("+") and x.split("-")[0][1:] in oses
            }
        ) - {x[1:] for x in options if x.startswith("-") and x.split("-")[1] in oses}

        test_dict = {os: [k for k in test_final if k.startswith(os)] for os in oses}
        build_dict = {
            os: [k for k in build if k.startswith(os) and len(test_dict[os]) > 0]
            for os in oses
        }

        if "ubuntu-latest" not in build_dict["ubuntu"] or "skip-doc" in options:
            build_doc = "false"

    print(f"::set-output name=build::{build_dict}")
    print(f"::set-output name=test::{test_dict}")
    print(f"::set-output name=build_doc::{build_doc}")
    for os in oses:
        print(
            f"::set-output name=do_{os}::{'true' if len(build_dict[os]) > 0 else 'false'}"
        )
    print(f"::set-output name=python_version::{python_options}")


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser(description="Setup the options")
    parser.add_argument(
        "--python-version",
        type=str,
        default="3.7,3.8,3.9",
        help="Python versions to build",
    )
    parser.add_argument(
        "--build",
        type=str,
        default="macos-10.15,ubuntu-latest,windows-2016",
        help="Python versions to build",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="macos-10.15,macos-11,ubuntu-18.04,ubuntu-20.04,windows-2016,windows-2019,windows-2022",
        help="Test options",
    )
    parser.add_argument("--message", type=str, help="The commit message")

    args = parser.parse_args()

    setup(
        args.message,
        python_version=args.python_version,
        build=args.build,
        test=args.test,
    )
