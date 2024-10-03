# pl-uae_gen

[![Version](https://img.shields.io/docker/v/fnndsc/pl-uae_gen?sort=semver)](https://hub.docker.com/r/fnndsc/pl-uae_gen)
[![MIT License](https://img.shields.io/github/license/fnndsc/pl-uae_gen)](https://github.com/FNNDSC/pl-uae_gen/blob/main/LICENSE)
[![ci](https://github.com/FNNDSC/pl-uae_gen/actions/workflows/ci.yml/badge.svg)](https://github.com/FNNDSC/pl-uae_gen/actions/workflows/ci.yml)

`pl-uae_gen` is a [_ChRIS_](https://chrisproject.org/)
_ds_ plugin which takes in ...  as input files and
creates ... as output files.

## Abstract

...

## Installation

`pl-uae_gen` is a _[ChRIS](https://chrisproject.org/) plugin_, meaning it can
run from either within _ChRIS_ or the command-line.

## Local Usage

To get started with local command-line usage, use [Apptainer](https://apptainer.org/)
(a.k.a. Singularity) to run `pl-uae_gen` as a container:

```shell
apptainer exec docker://fnndsc/pl-uae_gen uae_gen [--args values...] input/ output/
```

To print its available options, run:

```shell
apptainer exec docker://fnndsc/pl-uae_gen uae_gen --help
```

## Examples

`uae_gen` requires two positional arguments: a directory containing
input data, and a directory where to create output data.
First, create the input directory and move input data into it.

```shell
mkdir incoming/ outgoing/
mv some.dat other.dat incoming/
apptainer exec docker://fnndsc/pl-uae_gen:latest uae_gen [--args] incoming/ outgoing/
```

## Development

Instructions for developers.

### Building

Build a local container image:

```shell
docker build -t localhost/fnndsc/pl-uae_gen .
```

### Running

Mount the source code `uae_gen.py` into a container to try out changes without rebuild.

```shell
docker run --rm -it --userns=host -u $(id -u):$(id -g) \
    -v $PWD/uae_gen.py:/usr/local/lib/python3.12/site-packages/uae_gen.py:ro \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw -w /outgoing \
    localhost/fnndsc/pl-uae_gen uae_gen /incoming /outgoing
```

### Testing

Run unit tests using `pytest`.
It's recommended to rebuild the image to ensure that sources are up-to-date.
Use the option `--build-arg extras_require=dev` to install extra dependencies for testing.

```shell
docker build -t localhost/fnndsc/pl-uae_gen:dev --build-arg extras_require=dev .
docker run --rm -it localhost/fnndsc/pl-uae_gen:dev pytest
```

## Release

Steps for release can be automated by [Github Actions](.github/workflows/ci.yml).
This section is about how to do those steps manually.

### Increase Version Number

Increase the version number in `setup.py` and commit this file.

### Push Container Image

Build and push an image tagged by the version. For example, for version `1.2.3`:

```
docker build -t docker.io/fnndsc/pl-uae_gen:1.2.3 .
docker push docker.io/fnndsc/pl-uae_gen:1.2.3
```

### Get JSON Representation

Run [`chris_plugin_info`](https://github.com/FNNDSC/chris_plugin#usage)
to produce a JSON description of this plugin, which can be uploaded to _ChRIS_.

```shell
docker run --rm docker.io/fnndsc/pl-uae_gen:1.2.3 chris_plugin_info -d docker.io/fnndsc/pl-uae_gen:1.2.3 > chris_plugin_info.json
```

Intructions on how to upload the plugin to _ChRIS_ can be found here:
https://chrisproject.org/docs/tutorials/upload_plugin

