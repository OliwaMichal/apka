from typing import List, Dict
import html

def pick_type_tag(tags: List[str], subject: str = "") -> str:
    tags_u = [str(x).strip().upper() for x in (tags or [])]
    mapping = {
        "WYKŁAD": "WYKŁAD", "LECTURE": "WYKŁAD",
        "ĆWICZENIA": "ĆWICZENIA", "CWICZENIA": "ĆWICZENIA", "EXERCISE": "ĆWICZENIA",
        "LABORATORIUM": "LABORATORIUM", "LAB": "LABORATORIUM",
        "PROJEKT": "PROJEKT", "PROJECT": "PROJEKT",
        "SEMINARIUM": "SEMINARIUM",
    }
    for raw, out in mapping.items():
        if raw in tags_u:
            return out
    s = str(subject or "").upper()
    if " - W" in s: return "WYKŁAD"
    if " - C" in s or " - Ć" in s: return "ĆWICZENIA"
    if " - L" in s: return "LABORATORIUM"
    if " - P" in s: return "PROJEKT"
    return ""

def render_grid_html(
    days: List[str],
    hours: List[str],
    cell_map: Dict[Tuple[str, str], List[dict]],
    title: str,
) -> str:
    css = ... # bez zmian

    html = [css, f"<div class='tt-title'>{title}</div>", "<div class='tt-wrap'>", "<table class='tt'>"]
    html.append("<tr>")
    html.append("<th class='hour'>Godzina</th>")
    for d in days:
        html.append(f"<th>{d}</th>")
    html.append("</tr>")

    for h in hours:
        html.append("<tr>")
        html.append(f"<td class='hour'>{h}</td>")
        for d in days:
            key_str = f"{d}|{h}"
            acts = dedup_tiles(cell_map.get(key_str, cell_map.get((d, h), [])))
            cell = []
            for a in acts:
                # reszta bez zmian
                tags = a.get("tags") or []
                type_tag = pick_type_tag(tags, a.get("subject") or "")
                week_tag = ""
                tags_u = [str(x).strip().upper() for x in tags]
                if "ODD" in tags_u:
                    week_tag = "ODD"
                elif "EVEN" in tags_u:
                    week_tag = "EVEN"

                cls = type_tag if type_tag in {"WYKŁAD", "ĆWICZENIA", "LABORATORIUM", "PROJEKT", "SEMINARIUM"} else ""
                badges = []
                if type_tag:
                    badges.append(f"<span class='badge'>{type_tag}</span>")
                if week_tag:
                    badges.append(f"<span class='badge'>{week_tag}</span>")

                subj = str(a.get("subject") or "").strip() or "Zajęcia"
                teachers = ", ".join(a.get("teachers") or [])
                room = str(a.get("room") or "").strip()

                meta_parts = []
                if teachers:
                    meta_parts.append(f"Prow.: {teachers}")
                if room:
                    meta_parts.append(f"Sala: {room}")

                meta = "<br/>".join(meta_parts)

                cell.append(
                    f"<div class='tile {cls}'>"
                    f"<div class='subj'>{''.join(badges)}{subj}</div>"
                    f"<div class='meta'>{meta}</div>"
                    f"</div>"
                )
            html.append("<td>" + "".join(cell) + "</td>")
        html.append("</tr>")
    html.append("</table></div>")
    return "\n".join(html)