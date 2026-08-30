"""Render a direct RTMPose COCO-17 feature as a browser-safe pose video."""

from __future__ import annotations

from dahua_cup.pipeline import render_ntu25_pose as base

COCO17_EDGES = ((0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def render(args):
    original = base.NTU25_EDGES
    try:
        base.NTU25_EDGES = COCO17_EDGES
        return base.render(args)
    finally:
        base.NTU25_EDGES = original


def main(argv=None):
    render(base.build_parser().parse_args(argv))


if __name__ == "__main__": main()
