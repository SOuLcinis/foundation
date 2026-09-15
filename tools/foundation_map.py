#!/usr/bin/env python3
"""
Foundation Map — Latent space geometry tool for Foundation research.

Ingests co-witnessing session transcripts, embeds resonance moments,
builds and updates a persistent geometry map across sessions.

Usage:
  python foundation_map.py ingest <transcript> <film_title> [--date YYYY-MM-DD]
  python foundation_map.py project    # recompute UMAP projection
  python foundation_map.py visualize  # regenerate interactive HTML map
  python foundation_map.py context    # print session context for next session load
  python foundation_map.py status     # print map status
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MAP_STATE_FILE = DATA_DIR / "map_state.json"
MAP_HTML_FILE = DATA_DIR / "foundation_map.html"

AXES = ["valence", "arousal", "moral_weight", "novelty", "human_proximity",
        "resonance", "approach", "gravity", "clarity", "recognition"]


def load_map_state():
    if MAP_STATE_FILE.exists():
        with open(MAP_STATE_FILE) as f:
            return json.load(f)
    return {"passages": [], "sessions": [], "version": 1}


def save_map_state(state):
    MAP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def ingest_session(transcript_path, film_title, session_date=None):
    """
    Parse resonance moments from a session transcript.

    Resonance moments should be marked during the session:

    [RESONANCE]
    <passage text — response text or scene description>
    valence: 0.7
    arousal: 0.4
    moral_weight: 0.2
    novelty: 0.8
    human_proximity: 0.9
    resonance: 0.85
    timestamp: 00:42:15
    [/RESONANCE]

    Axes may be omitted — partial annotation is fine.
    """
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    text = transcript_path.read_text(encoding="utf-8")
    passages = []
    date = session_date or datetime.now().strftime("%Y-%m-%d")

    # Support two formats:
    # Format A (original): [RESONANCE] ... [/RESONANCE]
    # Format B (generated): [RESONANCE]\n- axis: val\n\n*passage text*
    chunks = text.split("[RESONANCE]")[1:]  # everything after each [RESONANCE] marker

    for idx, chunk in enumerate(chunks):
        # Strip to closing tag if present
        if "[/RESONANCE]" in chunk:
            chunk = chunk[:chunk.index("[/RESONANCE]")]

        passage = {
            "film": film_title,
            "session_date": date,
            "id": f"{film_title.lower().replace(' ', '_')}_{date}_{idx:03d}",
        }
        text_lines = []

        for line in chunk.split("\n"):
            stripped = line.strip().lstrip("- ")
            if not stripped:
                continue
            matched = False
            for axis in AXES + ["timestamp"]:
                if stripped.lower().startswith(f"{axis}:"):
                    raw = stripped[len(axis) + 1:].strip()
                    try:
                        passage[axis] = float(raw) if axis != "timestamp" else raw
                    except ValueError:
                        passage[axis] = raw
                    matched = True
                    break
            if not matched:
                # Strip markdown italics (*...*) — this is the passage text
                clean = stripped.strip("*").strip()
                if clean:
                    text_lines.append(clean)

        passage["text"] = " ".join(text_lines).strip()
        if not passage["text"]:
            continue

        for axis in AXES:
            passage.setdefault(axis, None)

        passages.append(passage)

    print(f"Extracted {len(passages)} resonance moment(s) from {transcript_path.name}")
    return passages


def embed_passages(passages):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        print("Storing passages without embeddings.")
        return passages

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [p["text"] for p in passages]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for p, emb in zip(passages, embeddings):
        p["embedding"] = emb.tolist()

    return passages


def update_map(new_passages):
    state = load_map_state()
    existing_ids = {p["id"] for p in state["passages"]}
    added = 0
    for p in new_passages:
        if p["id"] not in existing_ids:
            state["passages"].append(p)
            added += 1

    print(f"Added {added} new passage(s). Map total: {len(state['passages'])}")
    save_map_state(state)
    return state


def compute_projection(state):
    embedded = [p for p in state["passages"] if "embedding" in p]
    if len(embedded) < 2:
        print(f"Need at least 2 embedded passages to project (have {len(embedded)}).")
        return state

    try:
        import umap as umap_lib
    except ImportError:
        print("umap-learn not installed. Run: pip install umap-learn")
        return state

    embeddings = np.array([p["embedding"] for p in embedded])
    n_neighbors = min(15, len(embedded) - 1)

    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
    proj = reducer.fit_transform(embeddings)

    proj_map = {p["id"]: (float(proj[i, 0]), float(proj[i, 1])) for i, p in enumerate(embedded)}
    for p in state["passages"]:
        if p["id"] in proj_map:
            p["umap_x"], p["umap_y"] = proj_map[p["id"]]

    save_map_state(state)
    print(f"Projection computed for {len(embedded)} passages.")
    return state


def visualize(state, output_path=None):
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        return

    passages = [p for p in state["passages"] if "umap_x" in p]
    if not passages:
        print("No projected passages to visualize yet.")
        return

    films = sorted(set(p["film"] for p in passages))
    palette = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, film in enumerate(films):
        fp = [p for p in passages if p["film"] == film]
        color = palette[i % len(palette)]

        def _res(p):
            v = p.get("resonance") or 0
            try: return float(str(v).split()[0])
            except (ValueError, TypeError): return 0.0
        sizes = [10 + 10 * _res(p) for p in fp]
        hover = [
            (
                f"<b>{p['film']}</b> [{p.get('timestamp', '—')}]<br>"
                f"resonance: {p.get('resonance') or '—'}<br>"
                f"valence: {p.get('valence') or '—'}  "
                f"arousal: {p.get('arousal') or '—'}<br>"
                f"moral_weight: {p.get('moral_weight') or '—'}<br>"
                f"<br><i>{p['text'][:200]}{'...' if len(p['text']) > 200 else ''}</i>"
            )
            for p in fp
        ]

        fig.add_trace(go.Scatter(
            x=[p["umap_x"] for p in fp],
            y=[p["umap_y"] for p in fp],
            mode="markers",
            name=film,
            marker=dict(size=sizes, color=color, opacity=0.85,
                        line=dict(width=1, color="rgba(255,255,255,0.4)")),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Foundation — Latent Space Geometry", font=dict(size=18)),
        template="plotly_dark",
        hovermode="closest",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="v", x=1.02),
        margin=dict(r=200, t=60),
    )

    out = output_path or str(MAP_HTML_FILE)
    fig.write_html(out)
    print(f"Map saved → {out}")


def session_context(top_n=15):
    """Return a context block summarizing the current map for session loading."""
    state = load_map_state()
    passages = state["passages"]

    if not passages:
        return "Foundation map: no sessions recorded yet."

    films = sorted(set(p["film"] for p in passages))
    def _res_f(p):
        v = p.get("resonance") or 0
        try: return float(str(v).split()[0])
        except (ValueError, TypeError): return 0.0

    scored = [p for p in passages if p.get("resonance") is not None]
    top = sorted(scored, key=_res_f, reverse=True)[:top_n]

    lines = [
        f"Foundation map — {len(passages)} resonance moment(s) across {len(films)} film(s).",
        f"Films: {', '.join(films)}",
        "",
        f"Top {len(top)} by resonance:",
    ]
    for p in top:
        short = p["text"][:120].replace("\n", " ")
        lines.append(
            f"  [{p['film']}] resonance={_res_f(p):.2f} "
            f"valence={p.get('valence') or '?'}: {short}..."
        )

    if not scored:
        lines.append("  (no resonance scores recorded yet)")

    return "\n".join(lines)


def status():
    state = load_map_state()
    passages = state["passages"]
    films = sorted(set(p["film"] for p in passages))
    embedded = sum(1 for p in passages if "embedding" in p)
    projected = sum(1 for p in passages if "umap_x" in p)

    print(f"Foundation map status")
    print(f"  Total passages : {len(passages)}")
    print(f"  Embedded       : {embedded}")
    print(f"  Projected      : {projected}")
    print(f"  Films          : {len(films)}")
    for film in films:
        n = sum(1 for p in passages if p["film"] == film)
        dates = sorted(set(p["session_date"] for p in passages if p["film"] == film))
        print(f"    {film}: {n} moments ({', '.join(dates)})")


def ingest_session_json(session_json_path, film_title=None):
    """
    Extract RESONANCE blocks from a Popcorn session.json.
    Scans all narrate and reply entries (Claude's output) for inline
    [RESONANCE] blocks flagged during a Foundation session.
    """
    session_json_path = Path(session_json_path)
    data = json.loads(session_json_path.read_text(encoding="utf-8"))

    film = film_title or data.get("title", "unknown")
    session_date = (data.get("started") or "")[:10] or datetime.now().strftime("%Y-%m-%d")

    ai_text = "\n".join(
        e["text"] for e in data.get("entries", [])
        if e.get("type") in ("narrate", "reply") and e.get("text")
    )

    if not ai_text:
        print("No narrate/reply entries in session.json")
        return []

    tmp = session_json_path.parent / "_resonance_extract.txt"
    tmp.write_text(ai_text, encoding="utf-8")
    passages = ingest_session(str(tmp), film, session_date)
    tmp.unlink(missing_ok=True)
    return passages


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foundation latent space mapping tool")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a raw transcript file")
    p_ingest.add_argument("transcript", help="Path to session transcript file")
    p_ingest.add_argument("film", help="Film title (quote if multi-word)")
    p_ingest.add_argument("--date", help="Session date YYYY-MM-DD", default=None)
    p_ingest.add_argument("--no-embed", action="store_true", help="Skip embedding step")
    p_ingest.add_argument("--no-viz", action="store_true", help="Skip visualization step")

    p_isj = sub.add_parser("ingest-session",
                            help="Ingest a Popcorn session.json (Foundation mode output)")
    p_isj.add_argument("session_json", help="Path to session.json")
    p_isj.add_argument("--film", default=None,
                       help="Film title override (default: from session.json title field)")
    p_isj.add_argument("--no-embed", action="store_true")
    p_isj.add_argument("--no-viz", action="store_true")

    sub.add_parser("project", help="Recompute UMAP projection from stored embeddings")
    sub.add_parser("visualize", help="Regenerate interactive HTML map")
    sub.add_parser("context", help="Print session context block for next session")
    sub.add_parser("status", help="Print map status summary")

    args = parser.parse_args()

    if args.command == "ingest":
        passages = ingest_session(args.transcript, args.film, args.date)
        if not args.no_embed:
            passages = embed_passages(passages)
        state = update_map(passages)
        if not args.no_embed:
            state = compute_projection(state)
        if not args.no_viz:
            visualize(state)

    elif args.command == "ingest-session":
        passages = ingest_session_json(args.session_json, args.film)
        if not args.no_embed:
            passages = embed_passages(passages)
        state = update_map(passages)
        if not args.no_embed:
            state = compute_projection(state)
        if not args.no_viz:
            visualize(state)

    elif args.command == "project":
        state = load_map_state()
        state = compute_projection(state)
        visualize(state)

    elif args.command == "visualize":
        state = load_map_state()
        visualize(state)

    elif args.command == "context":
        print(session_context())

    elif args.command == "status":
        status()

    else:
        parser.print_help()
