# NBV Wildlife Detection

Research prototype for next-best-view (NBV) planning for UAV wildlife search in a procedural forest.

## Integrated sequential wildlife-aware experiment

This experiment starts a simulated drone near a forest corner and repeatedly evaluates locally reachable camera poses. It combines:
- incremental forest-mesh vertex visibility (geometry coverage proxy),
- incremental visibility of sampled surfaces on simplified wildlife box targets,
- target-observability threshold crossings (currently a configurable proxy),
- movement, rotation, and revisit penalties.

The target boxes are **simulation proxies**, not animal models. The experiment measures geometric observability only. It does not establish an improvement in real wildlife detector accuracy. Target proxy surfaces are not added to the forest raycasting occluder scene.

### Setup (Windows CMD, from repository root)

```cmd
py -m venv .venv
.venv\Scripts\activate
py -m pip install -e .
py -m pip install -r requirements.txt
```

### Run

```cmd
python experiments\run_wildlife_exploration.py --steps 20
```

Options:
- `--seed`: forest random seed
- `--target-seed`: target placement seed
- `--steps`: maximum sequential moves (the starting observation is additional)
- `--step-size`: nominal local movement distance in metres
- `--targets`: number of proxy targets
- `--connectivity`: 4 or 8 neighbor movement
- `--geometry-weight`, `--wildlife-weight`: normalized multi-objective gain weights
- `--threshold`: fraction of a target's sampled surface required to call it proxy-observable
- `--output`: output folder

Example:

```cmd
python experiments\run_wildlife_exploration.py --steps 25 --seed 42 --target-seed 123 --connectivity 8 --geometry-weight 0.45 --wildlife-weight 0.55
```

The runner saves `run_summary.json` and `step_history.csv` under the output directory. Seeds and experiment parameters are included in the JSON for repeatability.

## Geometry-only baseline

The earlier sequential geometry-only baseline remains available:

```cmd
python experiments\run_local_exploration.py --steps 15
```

Keep baseline and wildlife-aware runs separate when reporting results. For research comparisons, run multiple forest/target seeds and report distributions; do not draw conclusions from one run.

## Current limitations / next research steps

- Simplified axis-aligned boxes are not realistic animal shapes or poses.
- Random target placement currently does not reject tree/target or target/target overlap.
- Fixed altitude and simplified camera orientations/motion are used.
- Weighted objective values and threshold are experimental settings, not learned or validated parameters.
- Proxy observability is not detector recall/precision. A later evaluation must use actual detector predictions on images (or a defensible rendered dataset), with matched baseline conditions and metrics.
