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
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from shutil import rmtree
from subprocess import call
import sys
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def throttled(once_every):
    last_time = [0]
    def decorator(f):
        def decorated(*args, **kwargs):
            if time.time() - last_time[0] > once_every:
                last_time[0] = time.time()
                return f(*args, **kwargs)
        return decorated
    return decorator


@throttled(3)
def build_docs(args):
    # Remove the directory
    rmtree(args.output_dir)
    call(["mkdocs", "build", "-d", args.output_dir])
    call(["pdoc", "--html", "-o", os.path.join(args.output_dir, "api_docs"),
          "fast_transformers"])


def serve_docs(args):
    class BuildDocsEventHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            if os.path.splitext(event.src_path)[1] in [".md", ".py"]:
                build_docs(args)

    build_docs(args)
    this_dir = os.path.dirname(os.path.realpath(__file__))
    observer = Observer()
    observer.schedule(BuildDocsEventHandler(), this_dir, recursive=True)
    observer.start()
    try:
        handler = partial(SimpleHTTPRequestHandler, directory=args.output_dir)
        httpd = HTTPServer(args.bind, handler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


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

    # Serve the documentation (for writing the docs)
    serve = subparsers.add_parser(
        "serve_docs",
        help="Serve the documentation site for development purposes"
    )
    serve.add_argument(
        "--bind", "-b",
        type=lambda x: (x.split(":")[0], int(x.split(":")[1])),
        default=("", 8000),
        help="The address and port to bind the server to (default: :8000)"
    )
    serve.add_argument(
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
        build_docs=build_docs,
        serve_docs=serve_docs
    )[args.command](args)
