# VLA-EP

This workspace pulls together three related projects as git submodules:

- `open-pi-zero`
- `openvla-mini`
- `openvla-probe`

## Getting started

If you are cloning this repository for the first time, fetch the submodules at the same time:

```bash
git clone --recurse-submodules <repo-url>
cd VLA-EP
```

If you already cloned the repository, initialize the submodules with:

```bash
git submodule update --init --recursive
```

## Install the projects

Each submodule has its own environment and dependency instructions. A good default workflow is:

```bash
cd open-pi-zero
# follow the project README or install its Python dependencies

cd ../openvla-mini
# follow the project README or install its Python dependencies

cd ../openvla-probe
# follow the project README or install its Python dependencies
```

If you want to update every submodule to the latest recorded commit after pulling changes in this repo, run:

```bash
git submodule update --init --recursive
git submodule status
```
