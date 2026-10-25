from ai_platform.runtime.engine import get_app_graph
from ai_platform.core.query.planner import build_query_plan, plan_to_frame
from ai_platform.core.retrieval.ranking import (
    DEFAULT_RANKING_CONFIG, _lexical_score, _phrase_score,
    _rrf_signal, _semantic_signal, _quality_signal, _source_scope_signal,
    _attribute_alignment, _normalize, _effective_query_target, _explicit_target_match,
    detect_query_scopes, _is_generic_source, _is_specialized_source, _source, _document_text,
    _qualifier_match
)

q = "When does the postgraduate admission cycle typically occur for Semester 1 and Semester 2 at IIT Jodhpur?"
graph = get_app_graph()
# Run retrieval part
state = graph.invoke({'question': q, 'chat_history': ()})
fused = state.get('fused_candidates', ())
plan = build_query_plan(q)
frame = plan_to_frame(plan)

ranked = state.get('ranked_candidates', ())
top_ranked = ranked[0]
gt_ranked = None
for c in ranked:
    txt = getattr(c.document, "page_content", "")
    if "April-May" in txt or "November-December" in txt:
        gt_ranked = c
        break

print(f"Top Ranked: {getattr(top_ranked, 'source', '')} score={getattr(top_ranked, 'final_score', 0)}")
print(f"GT Ranked: {getattr(gt_ranked, 'source', '')} score={getattr(gt_ranked, 'final_score', 0)}")

config = DEFAULT_RANKING_CONFIG

for label, cand in [("Top Ranked", top_ranked), ("GT Ranked", gt_ranked)]:
    print(f"\n=== {label} ===")
    print("Source:", _source(cand))
    lex = config.lexical_weight * _lexical_score(q, cand, fused)
    phrase = config.phrase_weight * _phrase_score(q, cand, config.max_phrase_bonus)
    rrf = config.rrf_weight * _rrf_signal(cand)
    sem, cov = _semantic_signal(cand)
    sem_score = config.semantic_weight * sem + config.semantic_coverage_weight * cov
    
    attr_score, attrs = _attribute_alignment(q, cand, frame)
    attr_val = config.attribute_match_weight * attr_score if attrs else 0.0
    
    scope_match, scope_conflict = _source_scope_signal(q, cand)
    scope_val = config.scope_match_weight * scope_match - config.scope_conflict_penalty * scope_conflict
    
    qual_match, qual_miss = _qualifier_match(frame, cand)
    qual_val = config.qualifier_match_weight * qual_match - config.qualifier_miss_penalty * qual_miss
    
    quality_val = config.intrinsic_quality_weight * _quality_signal(cand)
    
    total = lex + phrase + rrf + sem_score + attr_val + scope_val + qual_val + quality_val
    print(f"  lexical: {lex:.3f}")
    print(f"  phrase: {phrase:.3f}")
    print(f"  rrf: {rrf:.3f}")
    print(f"  semantic: {sem_score:.3f} (sem={sem:.2f}, cov={cov:.2f})")
    print(f"  attribute: {attr_val:.3f} (attrs={attrs}, score={attr_score:.2f})")
    print(f"  scope: {scope_val:.3f} (match={scope_match:.2f}, conf={scope_conflict:.2f})")
    print(f"  qualifier: {qual_val:.3f}")
    print(f"  quality: {quality_val:.3f}")
    print(f"  Total approx: {total:.3f}")
