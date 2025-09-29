import argparse
import json
from typing import Any, Dict


def _load_scene(path: str) -> Dict[str, Any]:
    import yaml  # type: ignore
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="orcwa", description="OpenRCWA CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_sim = sub.add_parser("sim", help="Run a simple simulation from a YAML scene spec")
    p_sim.add_argument("scene", help="Path to scene YAML file")
    p_sim.add_argument("--out", help="Output JSON file (ResultGrid dataframe-like)")

    args = parser.parse_args(argv)

    if args.cmd == "sim":
        from rcwa.solve.simulate import simulate
        from rcwa.model.layer import Layer, LayerStack
        from rcwa.solve.source import LCP, RCP
        scene = _load_scene(args.scene)
        # Minimal schema: two half-spaces + one layer
        sup = Layer(er=scene.get('superstrate', {}).get('er', 1.0), ur=1.0)
        sub = Layer(er=scene.get('substrate', {}).get('er', 1.0), ur=1.0)
        layers = [Layer(er=l.get('er', 1.0), ur=l.get('ur', 1.0), thickness=l.get('thickness', 0.0)) for l in scene.get('layers', [])]
        stack = LayerStack(*layers, incident_layer=sup, transmission_layer=sub)
        wl = scene.get('wavelength', 0.5)
        th = scene.get('theta', 0.0)
        ph = scene.get('phi', 0.0)
        pol = scene.get('polarization', None)
        nh = scene.get('n_harmonics', 1)
        grid = simulate(stack, wl, theta=th, phi=ph, polarization=pol, n_harmonics=nh)
        # Emit a simple JSON: list of rows with RTot/TTot and coords
        df_like = grid.to_dataframe() if hasattr(grid, 'to_dataframe') else []
        if hasattr(df_like, 'to_dict'):
            payload = df_like.to_dict(orient='records')
        else:
            payload = df_like
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        else:
            print(json.dumps(payload, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
