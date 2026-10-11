#!/usr/bin/env bash
# Fetch the real-robot models used by sim/render3d.py (MuJoCo Menagerie, sparse checkout, ~250 MB).
set -euo pipefail
dest="$(dirname "$0")/../sim/assets/menagerie"
[ -d "$dest/.git" ] || git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$dest"
git -C "$dest" sparse-checkout set skydio_x2 unitree_go2 boston_dynamics_spot anybotics_anymal_c unitree_g1 unitree_h1
echo "menagerie ready at $dest"
