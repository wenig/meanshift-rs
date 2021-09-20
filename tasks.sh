#!/usr/bin/env bash

function test-build {
  maturin build -o wheels -i $(which python)
}

function test-install {
  test-build
  cd wheels && pip install --force-reinstall -U meanshift_rs-*.whl && cd ..
}

function release-build {
  maturin build --release -o wheels -i $(which python)
}

function release-install {
  release-build
  cd wheels && pip install --force-reinstall -U meanshift_rs-*.whl && cd ..
  install-py-meanshift
}

function build-run-tests {
  test-install
  pytest tests
}

function test {
  maturin develop
  pytest tests
}

function install-py-meanshift {
  pip install .
}

"$@"
