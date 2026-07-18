"""
Standalone subprocess helper for VeriEQL SQL equivalence checking.

Reads a JSON payload from stdin, runs VeriEQL, and writes a JSON result to stdout.
Designed to be invoked as a subprocess so that VeriEQL's Z3 state is fully isolated.

Input JSON (stdin):
    {
        "schema":    {"TABLE": {"col": "TYPE", ...}, ...},
        "pred_sql":  "<candidate SQL>",
        "gold_sql":  "<reference SQL>",
        "bound_size": 2          # optional, default 2
    }

Output JSON (stdout):
    {"result": true}              -- formally equivalent
    {"result": false}             -- provably non-equivalent
    {"result": null, "error": ".."} -- unknown / unsupported / parse error
"""

import json
import os
import sys


def _ensure_verieql_importable() -> None:
    """Make VeriEQL's flat modules importable.

    VeriEQL is installed as a git dependency (verieql @ git+...) whose build
    backend (hatchling) may not detect the flat .py files as installable modules.
    We find the actual source location via importlib.metadata and add it to sys.path.
    """
    try:
        import environment  # noqa: F401 – already importable, nothing to do
        return
    except ImportError:
        pass

    # Locate VeriEQL's source via its installed distribution metadata.
    import importlib.metadata
    import pathlib

    try:
        dist = importlib.metadata.distribution("verieql")

        # 1) Check if environment.py is listed in the installed files
        #    (happens when hatchling/setuptools DID install the flat modules).
        if dist.files:
            for f in dist.files:
                if f.name == "environment.py":
                    src_dir = str(f.locate().parent.resolve())
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)
                    return

        # 2) For editable / VCS installs, direct_url.json may contain a local file:// URL.
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            info = json.loads(direct_url_text)
            url = info.get("url", "")
            if url.startswith("file://"):
                src_dir = url[len("file://"):]
                if pathlib.Path(src_dir, "environment.py").exists():
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)
                    return

    except Exception:
        pass

    # 3) Last resort: honour an explicit env-var override.
    override = os.environ.get("VERIEQL_PATH", "")
    if override and override not in sys.path:
        sys.path.insert(0, override)


def main() -> None:
    data = json.loads(sys.stdin.read())
    schema: dict = data["schema"]
    pred_sql: str = data["pred_sql"]
    gold_sql: str = data["gold_sql"]
    bound_size: int = data.get("bound_size", 2)

    _ensure_verieql_importable()

    # Change to the VeriEQL source directory so its internal relative file
    # operations (e.g. loading grammar files) resolve correctly.
    try:
        import environment as _env_mod
        verieql_dir = os.path.dirname(os.path.abspath(_env_mod.__file__))
        os.chdir(verieql_dir)
    except Exception:
        pass

    try:
        from environment import Environment  # type: ignore[import]

        with Environment(generate_code=False, timer=False, show_counterexample=False) as env:
            for table_name, columns in schema.items():
                env.create_database(
                    attributes=columns,
                    bound_size=bound_size,
                    name=table_name,
                )
            env.save_checkpoints()
            result = env.analyze(pred_sql, gold_sql)

        if isinstance(result, bool):
            print(json.dumps({"result": result}))
        else:
            # result == -1 means column-count mismatch → not equivalent
            print(json.dumps({"result": False, "error": f"column mismatch (result={result})"}))

    except Exception as e:
        print(json.dumps({"result": None, "error": type(e).__name__ + ": " + str(e)}))


if __name__ == "__main__":
    main()
