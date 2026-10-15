"""``foxmesh-ai`` command line.

    foxmesh-ai dataset synth     DATASET --count 2000           # procedural bootstrap data
    foxmesh-ai dataset import    DATASET FOLDER --license CC-BY-4.0
    foxmesh-ai dataset stats     DATASET
    foxmesh-ai dataset validate  DATASET [--deep]
    foxmesh-ai dataset split     DATASET --ratios 0.9 0.05 0.05
    foxmesh-ai dataset preprocess DATASET [--workers 4]

    foxmesh-ai train     DATASET --out RUN [--preset small|base] [--epochs 60] [--resume]
    foxmesh-ai finetune  DATASET --out RUN --store MODELS --from v1
    foxmesh-ai eval      DATASET --store MODELS --model v1 [--modality text|image]
    foxmesh-ai generate  --store MODELS --model v1 --prompt "빨간 의자" [--image photo.png] --out chair.glb

    foxmesh-ai models list    --store MODELS
    foxmesh-ai models register --store MODELS RUN/checkpoints/best.pt [--eval DATASET] [--promote]
    foxmesh-ai models promote --store MODELS v2
    foxmesh-ai models export  --store MODELS --to PLATFORM_MODELS_DIR   # auto-selection engine
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


def _progress(label: str):
    def report(done: int, total: int) -> None:
        if done == total or done % max(total // 20, 1) == 0:
            print(f"\r{label}: {done}/{total}", end="\n" if done == total else "", file=sys.stderr, flush=True)
    return report


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# -- dataset ---------------------------------------------------------------------


def cmd_dataset(args) -> int:
    from .data.dataset import Dataset
    from .data.preprocess import PreprocessConfig, preprocess_dataset

    if args.action == "synth":
        from .data.synthetic import build_synthetic_dataset

        ds, added = build_synthetic_dataset(args.dataset, args.count, seed=args.seed,
                                            families=args.families, progress=_progress("generating"))
        _print({"added": added, **ds.stats()})
        return 0
    if args.action == "create":
        ds = Dataset.create(args.dataset, args.name or Path(args.dataset).name, args.description or "",
                            args.license or "mixed")
        _print(ds.stats())
        return 0
    ds = Dataset(args.dataset)
    if args.action == "import":
        counts = ds.import_folder(args.folder, source=args.source, license=args.license or "unknown",
                                  tags=args.tag or [])
        _print({**counts, "items": len(ds)})
    elif args.action == "stats":
        _print(ds.stats())
    elif args.action == "validate":
        report = ds.validate(deep=args.deep)
        _print({"ok": report.ok, "items": report.items, "errors": report.errors,
                "warnings": report.warnings[:50], "warning_count": len(report.warnings)})
        return 0 if report.ok else 1
    elif args.action == "split":
        _print(ds.resplit(tuple(args.ratios), args.seed))
    elif args.action == "preprocess":
        cfg = PreprocessConfig(render_size=args.render_size, grid=args.grid)
        result = preprocess_dataset(ds, cfg, workers=args.workers, force=args.force,
                                    progress=_progress("preprocessing"))
        _print(result)
        return 0 if not result["errors"] else 1
    return 0


# -- training --------------------------------------------------------------------


def _train_config(args, base=None):
    from .train import TrainConfig, finetune_config

    cfg = finetune_config() if base == "finetune" else TrainConfig()
    for f in fields(cfg):
        value = getattr(args, f.name, None)
        if value is not None:
            setattr(cfg, f.name, value)
    return cfg


def cmd_train(args) -> int:
    from .data.dataset import Dataset
    from .model import ModelConfig
    from .train import Trainer

    ds = Dataset(args.dataset)
    trainer = Trainer(ds, args.out, _train_config(args), ModelConfig.preset(args.preset), resume=args.resume)
    result = trainer.train()
    _print({k: v for k, v in result.__dict__.items() if k != "history"})
    return 0


def cmd_finetune(args) -> int:
    from .data.dataset import Dataset
    from .store import ModelStore
    from .train import Trainer

    store = ModelStore(args.store)
    base = store.checkpoint(args.base)
    cfg = _train_config(args, base="finetune")
    result = Trainer(Dataset(args.dataset), args.out, cfg, init_from=base).train()
    _print({k: v for k, v in result.__dict__.items() if k != "history"})
    return 0


def _evaluate_all(store, version: str, dataset_path: str, modalities, max_items, resolution) -> dict:
    from .data.dataset import Dataset
    from .evaluate import evaluate
    from .generate import Generator

    gen = Generator(store.checkpoint(version))
    ds = Dataset(dataset_path)
    split = "val" if ds.items("val") else "train"
    return {f"{split}_{m}": evaluate(gen, ds, split, m, resolution=resolution, max_items=max_items).to_dict()
            for m in modalities}


def cmd_eval(args) -> int:
    from .store import ModelStore

    store = ModelStore(args.store)
    version = store.resolve(args.model)
    metrics = _evaluate_all(store, version, args.dataset, args.modality, args.max_items, args.resolution)
    if args.save:
        store.set_metrics(version, {**store.index["versions"][version].get("metrics", {}), **metrics})
    _print(metrics)
    return 0


def cmd_generate(args) -> int:
    from .generate import Generator
    from .geometry import write_mesh
    from .store import ModelStore

    path = Path(args.checkpoint) if args.checkpoint else ModelStore(args.store).checkpoint(args.model)
    mesh = Generator(path).generate(args.prompt, args.image, resolution=args.resolution, size_m=args.size)
    write_mesh(mesh, args.out)
    _print({"out": args.out, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "watertight": mesh.watertight, **mesh.meta})
    return 0


def cmd_models(args) -> int:
    from .store import ModelStore

    store = ModelStore(args.store)
    if args.action == "list":
        prod = store.index["production"]
        for name, info in store.versions():
            metrics = info.get("metrics") or {}
            text = next((r for k, r in metrics.items() if k.endswith("_text")), {})
            print(f"{name:<6} {'*' if name == prod else ' '} params={info['parameters']:,} "
                  f"steps={info.get('steps')} dataset={(info.get('dataset') or {}).get('name')} "
                  f"iou={text.get('iou', '-')} fscore={text.get('fscore', '-')}")
    elif args.action == "register":
        version = store.register(args.checkpoint, notes=args.notes or "", parent=args.parent)
        if args.eval:
            store.set_metrics(version, _evaluate_all(store, version, args.eval, ["text", "image"],
                                                     args.max_items, args.resolution))
        if args.promote:
            store.promote(version)
        _print({"version": version, **store.index["versions"][version]})
    elif args.action == "promote":
        store.promote(args.version)
        _print({"production": args.version})
    elif args.action == "card":
        print(store.model_card(store.resolve(args.version)))
    elif args.action == "export":
        path = store.export(args.to, args.version, resolution=args.resolution)
        _print({"written": str(path), **store.platform_spec(args.version, resolution=args.resolution)})
    return 0


# -- parser ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foxmesh-ai", description="FoxMesh model: data, training, versions")
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("dataset", help="build and manage training datasets")
    ds_sub = ds.add_subparsers(dest="action", required=True)
    p = ds_sub.add_parser("synth", help="generate procedural shapes with captions")
    p.add_argument("dataset")
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", nargs="*")
    p = ds_sub.add_parser("create")
    p.add_argument("dataset")
    p.add_argument("--name")
    p.add_argument("--description")
    p.add_argument("--license")
    p = ds_sub.add_parser("import", help="import meshes (+ .txt/.json captions, images) from a folder")
    p.add_argument("dataset")
    p.add_argument("folder")
    p.add_argument("--source", default="import")
    p.add_argument("--license")
    p.add_argument("--tag", action="append")
    for name in ("stats", "validate", "split", "preprocess"):
        p = ds_sub.add_parser(name)
        p.add_argument("dataset")
        if name == "validate":
            p.add_argument("--deep", action="store_true")
        if name == "split":
            p.add_argument("--ratios", type=float, nargs=3, default=[0.9, 0.05, 0.05])
            p.add_argument("--seed", type=int, default=0)
        if name == "preprocess":
            p.add_argument("--workers", type=int)
            p.add_argument("--force", action="store_true")
            p.add_argument("--render-size", type=int, default=64)
            p.add_argument("--grid", type=int, default=128)
    ds.set_defaults(func=cmd_dataset)

    def train_args(p):
        p.add_argument("dataset")
        p.add_argument("--out", required=True)
        for name, kind in (("epochs", int), ("batch_size", int), ("lr", float), ("warmup_steps", int),
                           ("uniform_points", int), ("near_points", int), ("surface_points", int),
                           ("early_stop_patience", int), ("num_workers", int), ("seed", int),
                           ("max_steps", int)):
            p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind)
        p.add_argument("--device")
        p.add_argument("--freeze", nargs="*", choices=["text", "image", "decoder"])

    p = sub.add_parser("train", help="train a model from scratch")
    train_args(p)
    p.add_argument("--preset", default="small", choices=["small", "base"])
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("finetune", help="continue training a registered version on (new) data")
    train_args(p)
    p.add_argument("--store", required=True)
    p.add_argument("--from", dest="base", default="production")
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("eval", help="measure generation quality on a dataset split")
    p.add_argument("dataset")
    p.add_argument("--store", required=True)
    p.add_argument("--model", default="production")
    p.add_argument("--modality", nargs="+", default=["text", "image"], choices=["text", "image"])
    p.add_argument("--max-items", type=int)
    p.add_argument("--resolution", type=int, default=48)
    p.add_argument("--save", action="store_true", help="store the results with the version")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("generate", help="generate a mesh from a prompt and/or image")
    p.add_argument("--store")
    p.add_argument("--model", default="production")
    p.add_argument("--checkpoint", help="use a checkpoint file instead of the store")
    p.add_argument("--prompt")
    p.add_argument("--image")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--size", type=float, default=1.0, help="largest side in metres")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("models", help="model versions")
    m_sub = p.add_subparsers(dest="action", required=True)
    for name in ("list", "register", "promote", "card", "export"):
        q = m_sub.add_parser(name)
        q.add_argument("--store", required=True)
        if name == "register":
            q.add_argument("checkpoint")
            q.add_argument("--eval", metavar="DATASET", help="evaluate on this dataset and store metrics")
            q.add_argument("--max-items", type=int)
            q.add_argument("--resolution", type=int, default=48)
            q.add_argument("--promote", action="store_true")
            q.add_argument("--notes")
            q.add_argument("--parent")
        if name in ("promote", "card"):
            q.add_argument("version")
        if name == "export":
            q.add_argument("--to", required=True, help="platform models directory (FOXMESH_MODELS_DIR)")
            q.add_argument("--version", default="production")
            q.add_argument("--resolution", type=int, default=64)
    p.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
