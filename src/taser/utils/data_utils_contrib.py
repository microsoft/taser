# Copyright (c) 2023 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import os
import zipfile
from typing import List

import ujson

logger = logging.getLogger()


def read_data_from_jsonl_files(paths: List[str]) -> List:
    def _read_jsonl_file(fop):
        return [ujson.loads(line) for line in fop]

    if not paths:
        raise ValueError("not file to read!")

    results = []
    for i, path in enumerate(paths):
        logger.info(f"paths to be read {paths}")
        if os.path.splitext(path)[1] == ".zip":
            with zipfile.ZipFile(path) as fzip:
                input_fn_list = fzip.namelist()
                for input_fn in input_fn_list:
                    logger.info("Reading file %s" % input_fn)
                    with fzip.open(input_fn) as fin:
                        data = _read_jsonl_file(fin)
                    break
            results = data
        else:
            logger.info("Reading file %s" % path)
            with open(path, "r", encoding="utf-8") as f:
                data = _read_jsonl_file(f)
            results = data
        logger.info("Aggregated data size: {}".format(len(results)))
    return results
