from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLASS_NAMES = {
    0: 'chair',
    1: 'table',
    2: 'sofa',
    3: 'bed',
    4: 'cabinet',
    5: 'wardrobe',
    6: 'desk',
    7: 'bookshelf',
    8: 'nightstand',
    9: 'tv_stand',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze hidden/room coverage for exported real single-view scene packets.')
    parser.add_argument('--view-dir', required=True)
    parser.add_argument('--index-file', default='')
    parser.add_argument('--save-summary', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    return parser.parse_args()


def load_split_map(index_file: str | Path) -> dict[str, str]:
    if not index_file:
        return {}
    path = Path(index_file)
    if not path.exists():
        return {}
    sample_to_split: dict[str, str] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get('sample_id', ''))
            split = str(row.get('split', 'unknown'))
            if sample_id:
                sample_to_split[sample_id] = split
    return sample_to_split


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _counter_named(counter: Counter[int]) -> dict[str, int]:
    named: dict[str, int] = {}
    for cls_id, count in sorted(counter.items()):
        label = CLASS_NAMES.get(int(cls_id), f'class_{cls_id}')
        named[label] = int(count)
    return named


def main() -> None:
    args = parse_args()
    view_dir = Path(args.view_dir)
    if not view_dir.exists():
        raise FileNotFoundError(f'view dir does not exist: {view_dir}')

    sample_to_split = load_split_map(args.index_file)
    view_paths = sorted(view_dir.glob('*.json'))
    if not view_paths:
        raise RuntimeError(f'no view json files found in {view_dir}')

    records: list[dict[str, Any]] = []
    all_visible_uids: set[str] = set()
    all_hidden_uids: set[str] = set()
    room_to_views: defaultdict[str, list[str]] = defaultdict(list)
    room_to_scene: dict[str, str] = {}
    room_to_unique_uids: defaultdict[str, set[str]] = defaultdict(set)
    split_to_views: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    split_to_rooms: defaultdict[str, set[str]] = defaultdict(set)
    visible_class_counter: Counter[int] = Counter()
    hidden_class_counter: Counter[int] = Counter()
    hidden_supported_class_counter: Counter[int] = Counter()
    hidden_unsupported_class_counter: Counter[int] = Counter()
    hidden_uid_to_class: dict[str, int] = {}
    hidden_uid_occurrences: Counter[str] = Counter()
    visible_uid_occurrences: Counter[str] = Counter()

    for path in view_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        sample_id = str(payload['sample_id'])
        room_id = str(payload['room_id'])
        scene_id = str(payload['scene_id'])
        split = sample_to_split.get(sample_id, 'unknown')
        visible = payload.get('visible_objects', [])
        hidden = payload.get('hidden_objects', [])
        visible_uids = [str(obj['uid']) for obj in visible]
        hidden_uids = [str(obj['uid']) for obj in hidden]

        for obj in visible:
            cls_id = int(obj['class_id'])
            visible_class_counter[cls_id] += 1
            visible_uid_occurrences[str(obj['uid'])] += 1
            all_visible_uids.add(str(obj['uid']))
        for obj in hidden:
            cls_id = int(obj['class_id'])
            hidden_class_counter[cls_id] += 1
            hidden_uid = str(obj['uid'])
            hidden_uid_to_class.setdefault(hidden_uid, cls_id)
            hidden_uid_occurrences[hidden_uid] += 1
            all_hidden_uids.add(hidden_uid)

        room_to_views[room_id].append(sample_id)
        room_to_scene[room_id] = scene_id
        room_to_unique_uids[room_id].update(visible_uids)
        room_to_unique_uids[room_id].update(hidden_uids)
        split_to_rooms[split].add(room_id)

        record = {
            'sample_id': sample_id,
            'scene_id': scene_id,
            'room_id': room_id,
            'split': split,
            'num_visible': len(visible),
            'num_hidden': len(hidden),
            'num_major': len(visible) + len(hidden),
            'visible_uids': visible_uids,
            'hidden_uids': hidden_uids,
        }
        records.append(record)
        split_to_views[split].append(record)

    hidden_supported_uids = {uid for uid in all_hidden_uids if uid in all_visible_uids}
    hidden_unsupported_uids = sorted(all_hidden_uids - all_visible_uids)

    hidden_supported_occurrences = 0
    hidden_total_occurrences = 0
    split_supported_occurrences: defaultdict[str, int] = defaultdict(int)
    split_hidden_occurrences: defaultdict[str, int] = defaultdict(int)

    for record in records:
        split = str(record['split'])
        for uid in record['hidden_uids']:
            cls_id = hidden_uid_to_class[uid]
            hidden_total_occurrences += 1
            split_hidden_occurrences[split] += 1
            if uid in hidden_supported_uids:
                hidden_supported_occurrences += 1
                split_supported_occurrences[split] += 1
                hidden_supported_class_counter[cls_id] += 1
            else:
                hidden_unsupported_class_counter[cls_id] += 1

    view_hidden_counts = [int(r['num_hidden']) for r in records]
    view_visible_counts = [int(r['num_visible']) for r in records]
    room_view_counts = [len(v) for v in room_to_views.values()]
    room_object_counts = [len(v) for v in room_to_unique_uids.values()]

    per_split: dict[str, Any] = {}
    for split, split_records in sorted(split_to_views.items()):
        vis_counts = [int(r['num_visible']) for r in split_records]
        hid_counts = [int(r['num_hidden']) for r in split_records]
        split_hidden_uids = {uid for r in split_records for uid in r['hidden_uids']}
        split_visible_uids = {uid for r in split_records for uid in r['visible_uids']}
        split_hidden_supported_uids = split_hidden_uids & all_visible_uids
        per_split[split] = {
            'num_views': len(split_records),
            'num_rooms': len(split_to_rooms[split]),
            'avg_visible_per_view': round(_safe_mean(vis_counts), 4),
            'avg_hidden_per_view': round(_safe_mean(hid_counts), 4),
            'median_visible_per_view': round(_safe_median(vis_counts), 4),
            'median_hidden_per_view': round(_safe_median(hid_counts), 4),
            'unique_visible_uids': len(split_visible_uids),
            'unique_hidden_uids': len(split_hidden_uids),
            'unique_hidden_uids_with_visible_support_anywhere': len(split_hidden_supported_uids),
            'unique_hidden_uid_support_ratio': round(len(split_hidden_supported_uids) / max(len(split_hidden_uids), 1), 4),
            'hidden_occurrence_support_ratio': round(split_supported_occurrences[split] / max(split_hidden_occurrences[split], 1), 4),
        }

    room_rank = sorted(
        (
            {
                'room_id': room_id,
                'scene_id': room_to_scene[room_id],
                'num_views': len(sample_ids),
                'unique_object_uids': len(room_to_unique_uids[room_id]),
            }
            for room_id, sample_ids in room_to_views.items()
        ),
        key=lambda item: (-int(item['num_views']), -int(item['unique_object_uids']), item['room_id']),
    )

    summary = {
        'view_dir': str(view_dir),
        'index_file': str(args.index_file) if args.index_file else '',
        'num_views': len(records),
        'num_rooms': len(room_to_views),
        'num_scenes': len({r['scene_id'] for r in records}),
        'per_view': {
            'avg_visible_per_view': round(_safe_mean(view_visible_counts), 4),
            'avg_hidden_per_view': round(_safe_mean(view_hidden_counts), 4),
            'median_visible_per_view': round(_safe_median(view_visible_counts), 4),
            'median_hidden_per_view': round(_safe_median(view_hidden_counts), 4),
            'max_visible_per_view': max(view_visible_counts),
            'max_hidden_per_view': max(view_hidden_counts),
        },
        'room_coverage': {
            'avg_views_per_room': round(_safe_mean(room_view_counts), 4),
            'median_views_per_room': round(_safe_median(room_view_counts), 4),
            'rooms_with_ge_2_views': sum(1 for n in room_view_counts if n >= 2),
            'rooms_with_ge_3_views': sum(1 for n in room_view_counts if n >= 3),
            'rooms_with_ge_4_views': sum(1 for n in room_view_counts if n >= 4),
            'avg_unique_objects_per_room': round(_safe_mean(room_object_counts), 4),
            'median_unique_objects_per_room': round(_safe_median(room_object_counts), 4),
            'top_rooms_by_view_count': room_rank[: max(int(args.top_k), 1)],
        },
        'object_coverage': {
            'unique_visible_uids': len(all_visible_uids),
            'unique_hidden_uids': len(all_hidden_uids),
            'unique_hidden_uids_with_visible_support_anywhere': len(hidden_supported_uids),
            'unique_hidden_uid_support_ratio': round(len(hidden_supported_uids) / max(len(all_hidden_uids), 1), 4),
            'hidden_occurrences_total': int(hidden_total_occurrences),
            'hidden_occurrences_with_visible_support_anywhere': int(hidden_supported_occurrences),
            'hidden_occurrence_support_ratio': round(hidden_supported_occurrences / max(hidden_total_occurrences, 1), 4),
            'hidden_uids_never_visible_anywhere_examples': hidden_unsupported_uids[: max(int(args.top_k), 1)],
        },
        'class_coverage': {
            'visible_occurrences': _counter_named(visible_class_counter),
            'hidden_occurrences': _counter_named(hidden_class_counter),
            'hidden_supported_occurrences': _counter_named(hidden_supported_class_counter),
            'hidden_unsupported_occurrences': _counter_named(hidden_unsupported_class_counter),
        },
        'splits': per_split,
    }

    save_path = Path(args.save_summary)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
