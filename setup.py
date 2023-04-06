# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import setuptools

cur_dir = osp.dirname(osp.realpath(__file__))
requirementPath = osp.join(cur_dir, "requirements.txt")
install_requires = []
with open(requirementPath) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="imitation_learning",
    version="0.1",
    author="Andrew Szot",
    author_email="",
    description="imitation_learning",
    url="",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
