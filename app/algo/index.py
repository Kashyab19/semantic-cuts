from typing import Dict, List


def reciprocal_rank_fusion(
    list_a: List[Dict], list_b: List[Dict], k: int = 60
) -> List[Dict]:
    """
    Merges two lists by RANK rather than raw score.
    Essential because CLIP scores (0.25) are lower than BGE scores (0.75).
    """
    fused_scores = {}

    # Helper to process a list
    def process_list(results_list):
        for rank, item in enumerate(results_list):
            # Unique ID for the specific moment in the specific video
            doc_id = f"{item['video_id']}_{int(item['timestamp'])}"

            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"item": item, "score": 0}

            # The RRF Formula
            fused_scores[doc_id]["score"] += 1 / (k + rank + 1)

    process_list(list_a)
    process_list(list_b)

    # Sort by new fused score
    results = [val["item"] for val in fused_scores.values()]
    for val in fused_scores.values():
        val["item"]["score"] = val["score"]

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def deduplicate_results(results: List[Dict], time_threshold: float = 5.0) -> List[Dict]:
    """
    Prevents clutter. If we have hits at 10s, 11s, and 12s, just keep 10s.
    """
    clean_results = []
    seen_timestamps = []  # simple list of (video_id, timestamp)

    for res in results:
        is_duplicate = False
        for vid, ts in seen_timestamps:
            if res["video_id"] == vid and abs(res["timestamp"] - ts) < time_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            clean_results.append(res)
            seen_timestamps.append((res["video_id"], res["timestamp"]))

    return clean_results
