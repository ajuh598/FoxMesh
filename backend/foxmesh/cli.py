"""Command line entry point.

    foxmesh serve [--host 0.0.0.0] [--port 8000]
    foxmesh analyze "게임에 사용할 중세풍 검을 만들어 줘"
    foxmesh run "3D 프린팅용 10cm 컵" [--profile print_3d]
    foxmesh models
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .classifier import analyze
from .router import NoCandidateError
from .schemas import JobRequest, Profile
from .services import build_services


def _request(args: argparse.Namespace) -> JobRequest:
    return JobRequest(prompt=args.prompt, profile=Profile(args.profile) if args.profile else None,
                      model_id=args.model)


def cmd_analyze(args: argparse.Namespace) -> int:
    services = build_services()
    spec = analyze(_request(args), commercial_use=services.settings.commercial_use)
    try:
        decision = services.router.route(spec)
        error = None
    except NoCandidateError as exc:
        decision, error = exc.decision, str(exc)
    print(json.dumps({"spec": spec.model_dump(mode="json"),
                      "routing": decision.model_dump(mode="json"), "error": error},
                     indent=2, ensure_ascii=False))
    return 1 if error else 0


def cmd_run(args: argparse.Namespace) -> int:
    async def _run() -> int:
        services = build_services()
        await services.jobs.start()
        try:
            job, _ = services.jobs.submit(_request(args))
            job = await services.jobs.wait(job.id)
        finally:
            await services.jobs.stop()
        for event in job.events:
            print(f"[{event.stage.value:>14}] {event.message}", file=sys.stderr)
        print(json.dumps({"id": job.id, "status": job.status.value, "error": job.error,
                          "result": job.result.model_dump(mode="json") if job.result else None},
                         indent=2, ensure_ascii=False))
        return 0 if job.result else 1
    return asyncio.run(_run())


def cmd_models(args: argparse.Namespace) -> int:
    services = build_services()
    for view in services.registry.views():
        status = "ready" if view.available else f"unavailable ({view.unavailable_reason})"
        print(f"{view.spec.id:<24} {view.spec.version:<8} {status}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foxmesh")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("analyze", cmd_analyze), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("prompt")
        p.add_argument("--profile", choices=[p.value for p in Profile])
        p.add_argument("--model", help="force a specific model id")
        p.set_defaults(func=func)
    sub.add_parser("models").set_defaults(func=cmd_models)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
