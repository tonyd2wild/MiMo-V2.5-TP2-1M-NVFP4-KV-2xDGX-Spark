#!/usr/bin/env bash
set -euo pipefail

echo "[fix-prometheus-instrumentator-router] Patching route-name helper for FastAPI router objects"

python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/prometheus_fastapi_instrumentator/routing.py")
text = path.read_text()

old = '''    for route in routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_name = route.path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _get_route_name(child_scope, route.routes, route_name)
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = route.path
'''

new = '''    for route in routes:
        route_path = getattr(route, "path", None)
        if route_path is None or not hasattr(route, "matches"):
            continue
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_name = route_path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _get_route_name(child_scope, route.routes, route_name)
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = route_path
'''

if old in text:
    path.write_text(text.replace(old, new))
    print(f"[fix-prometheus-instrumentator-router] patched {path}")
elif 'route_path = getattr(route, "path", None)' in text:
    print("[fix-prometheus-instrumentator-router] already patched")
else:
    raise SystemExit("[fix-prometheus-instrumentator-router] expected routing.py anchor not found")
PY
