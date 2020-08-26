#!/usr/bin/env python
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""A script to contain some package administration tools and automations such
as building the documentation.

Maybe this script should be a shell-script, maybe it shouldn't. :-)
"""

import argparse
import os
from shutil import rmtree
from subprocess import call
import sys


def build_docs(args):
    # Remove the directory
    rmtree(args.output_dir)
    call(["mkdocs", "build", "-d", args.output_dir])
    call(["pdoc", "--html", "-o", os.path.join(args.output_dir, "api_docs"),
          "fast_transformers"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the documentation site"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Documentation command
    docs = subparsers.add_parser(
        "build_docs",
        help="Build the documentation site"
    )
    docs.add_argument(
        "--output_dir", "-o",
        default="site",
        help="Choose the output directory to store the html (default: site)"
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch the command
    dict(
        build_docs=build_docs
    )[args.command](args)
