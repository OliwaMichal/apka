from typing import List, Dict, Tuple
import html


def dedup_tiles(activities):
    seen = set()
    result = []

    for a in activities:
        key = (
            a.get("subject", ""),
            a.get("room", ""),
            tuple(a.get("teachers", []))
        )

        if key not in seen:
            seen.add(key)
            result.append(a)

    return result


def pick_type_tag(tags: List[str], subject: str = "") -> str:
    tags_u = [str(x).strip().upper() for x in (tags or [])]

    mapping = {
        "WYKŁAD": "WYKŁAD",
        "LECTURE": "WYKŁAD",

        "ĆWICZENIA": "ĆWICZENIA",
        "CWICZENIA": "ĆWICZENIA",
        "EXERCISE": "ĆWICZENIA",

        "LABORATORIUM": "LABORATORIUM",
        "LAB": "LABORATORIUM",

        "PROJEKT": "PROJEKT",
        "PROJECT": "PROJEKT",

        "SEMINARIUM": "SEMINARIUM",
    }

    for raw, out in mapping.items():
        if raw in tags_u:
            return out

    s = str(subject or "").upper()

    if " - W" in s:
        return "WYKŁAD"

    if " - C" in s or " - Ć" in s:
        return "ĆWICZENIA"

    if " - L" in s:
        return "LABORATORIUM"

    if " - P" in s:
        return "PROJEKT"

    return ""


def render_grid_html(
    days: List[str],
    hours: List[str],
    cell_map: Dict,
    title: str,
) -> str:

    css = """
    <style>
        .tt-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .tt-wrap {
            overflow-x: auto;
            margin-bottom: 20px;
        }

        table.tt {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            font-size: 12px;
        }

        table.tt th,
        table.tt td {
            border: 1px solid #ccc;
            padding: 4px;
            vertical-align: top;
        }

        table.tt th {
            background: #f2f2f2;
            text-align: center;
        }

        .hour {
            width: 70px;
            background: #fafafa;
            font-weight: bold;
            text-align: center;
        }

        .tile {
            border-radius: 6px;
            padding: 6px;
            margin-bottom: 5px;
            color: #111;
            font-size: 11px;
        }

        .WYKŁAD {
            background: #dbeafe;
        }

        .ĆWICZENIA {
            background: #dcfce7;
        }

        .LABORATORIUM {
            background: #fef3c7;
        }

        .PROJEKT {
            background: #f3e8ff;
        }

        .SEMINARIUM {
            background: #ffe4e6;
        }

        .subj {
            font-weight: bold;
            margin-bottom: 4px;
        }

        .meta {
            font-size: 10px;
            color: #444;
        }

        .badge {
            display: inline-block;
            padding: 2px 5px;
            border-radius: 4px;
            background: #333;
            color: white;
            font-size: 9px;
            margin-right: 4px;
        }
    </style>
    """

    html_parts = [
        css,
        f"<div class='tt-title'>{html.escape(title)}</div>",
        "<div class='tt-wrap'>",
        "<table class='tt'>"
    ]

    html_parts.append("<tr>")
    html_parts.append("<th class='hour'>Godzina</th>")

    for d in days:
        html_parts.append(f"<th>{html.escape(d)}</th>")

    html_parts.append("</tr>")

    for h in hours:

        html_parts.append("<tr>")
        html_parts.append(f"<td class='hour'>{html.escape(h)}</td>")

        for d in days:

            key_str = f"{d}|{h}"

            acts = dedup_tiles(
                cell_map.get(
                    key_str,
                    cell_map.get((d, h), [])
                )
            )

            cell = []

            for a in acts:

                tags = a.get("tags") or []

                type_tag = pick_type_tag(
                    tags,
                    a.get("subject") or ""
                )

                week_tag = ""

                tags_u = [
                    str(x).strip().upper()
                    for x in tags
                ]

                if "ODD" in tags_u:
                    week_tag = "ODD"

                elif "EVEN" in tags_u:
                    week_tag = "EVEN"

                cls = type_tag if type_tag in {
                    "WYKŁAD",
                    "ĆWICZENIA",
                    "LABORATORIUM",
                    "PROJEKT",
                    "SEMINARIUM"
                } else ""

                badges = []

                if type_tag:
                    badges.append(
                        f"<span class='badge'>{html.escape(type_tag)}</span>"
                    )

                if week_tag:
                    badges.append(
                        f"<span class='badge'>{html.escape(week_tag)}</span>"
                    )

                subj = html.escape(
                    str(a.get("subject") or "").strip() or "Zajęcia"
                )

                teachers = ", ".join(
                    a.get("teachers") or []
                )

                room = str(a.get("room") or "").strip()

                meta_parts = []

                if teachers:
                    meta_parts.append(
                        f"Prow.: {html.escape(teachers)}"
                    )

                if room:
                    meta_parts.append(
                        f"Sala: {html.escape(room)}"
                    )

                meta = "<br/>".join(meta_parts)

                cell.append(
                    f"<div class='tile {cls}'>"
                    f"<div class='subj'>{''.join(badges)}{subj}</div>"
                    f"<div class='meta'>{meta}</div>"
                    f"</div>"
                )

            html_parts.append(
                "<td>" + "".join(cell) + "</td>"
            )

        html_parts.append("</tr>")

    html_parts.append("</table></div>")

    return "\n".join(html_parts)