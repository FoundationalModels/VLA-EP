# Shared conda bootstrap, sourced by the OpenVLA wrapper scripts.
#
# On this cluster `conda` is a shell function installed by `module load
# anaconda/...`, not a binary on PATH. Shell functions are not inherited by
# child processes, so a wrapper invoked as `bash scripts/foo.sh` — which is how
# the SLURM jobs call these — starts with no conda at all, even when the
# submitting shell had it. This bootstraps Lmod, loads the module, and activates
# the requested environment.
#
# Usage:
#   source "$SCRIPT_DIR/_activate_conda.sh" openvla-probe

_activate_conda() {
    local env_name="$1"
    local anaconda_module="anaconda/2025.06.0"

    if ! command -v module > /dev/null 2>&1; then
        # shellcheck disable=SC1091
        [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash
    fi

    if ! command -v conda > /dev/null 2>&1 && [[ -z "${CONDA_EXE:-}" ]]; then
        module load "$anaconda_module"
    fi

    # `conda shell.bash hook` needs the real binary, which the module exports as
    # CONDA_EXE even when `conda` itself is only a function.
    local conda_bin="${CONDA_EXE:-$(command -v conda)}"
    if [[ -z "$conda_bin" ]]; then
        echo "Error: could not locate conda after loading $anaconda_module" >&2
        return 1
    fi

    eval "$("$conda_bin" shell.bash hook)"
    conda activate "$env_name"
}
