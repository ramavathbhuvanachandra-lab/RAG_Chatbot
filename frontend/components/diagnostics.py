"""Retrieval Diagnostics and Developer Inspection Component for Streamlit Developer Mode.

Provides per-turn execution trace selection and independent expandable inspection panels:
  Panel 1: Query Understanding & Retrieval Handoff
  Panel 2: Dense Retrieval (Vector Search)
  Panel 3: BM25 / Lexical Retrieval
  Panel 4: Hybrid Fusion Results (RRF Top 15)
  Panel 5: Ranking and Verification Decisions
  Panel 6: Final Packaged Context & Prompt Messages
  Panel 7: Answer Generation Input and Output
  Panel 8: Stage Timings, Warnings, and Errors
"""

from __future__ import annotations

from typing import Any, Mapping
import streamlit as st


def render_developer_trace_inspector() -> None:
    """Render the centralized per-turn developer diagnostics inspector with trace selector."""
    if not st.session_state.get("show_retrieval_diagnostics", False):
        return

    traces = st.session_state.get("execution_traces", [])
    if not traces:
        with st.expander("🛠️ Developer Diagnostics — Trace Inspector", expanded=False):
            st.info(
                "No execution traces recorded yet in this session. "
                "Ask a question in chat to inspect query understanding, hybrid retrieval, ranking, and packaging."
            )
        return

    with st.expander("🛠️ Developer Diagnostics — Request and Evidence Trace", expanded=True):
        if st.session_state.get("_turn_in_progress", False):
            st.info("⏳ **Turn in Progress:** A new conversational turn is actively executing. Finalized stage measurements will appear here upon completion.")

        st.markdown("#### 🔬 Execution Trace Selector")
        st.caption(
            "Select any completed conversation turn to inspect its end-to-end retrieval and verification trace. "
            "Diagnostics are read-only and captured non-invasively during original execution."
        )

        trace_ids = [t["turn_id"] for t in traces]
        trace_lookup = {t["turn_id"]: t for t in traces}

        selected_id = st.session_state.get("selected_trace_id")
        if selected_id not in trace_lookup:
            selected_id = trace_ids[-1]
            st.session_state["selected_trace_id"] = selected_id

        def _format_trace(tid: str) -> str:
            t = trace_lookup[tid]
            turn_no = t.get("turn_number", 1)
            time_str = t.get("timestamp", "")
            preview = t.get("query_preview", "")
            latency = t.get("latency_seconds", 0.0)
            status = str(t.get("terminal_status", "SUCCESS")).upper()
            return f"Turn #{turn_no} [{time_str}] — \"{preview}\" ({latency:.2f}s) [{status}]"

        chosen_id = st.selectbox(
            "Select Turn Trace to Inspect:",
            options=trace_ids,
            index=trace_ids.index(selected_id),
            format_func=_format_trace,
            key="trace_selector_dropdown",
        )
        st.session_state["selected_trace_id"] = chosen_id

        active_trace = trace_lookup[chosen_id]
        diag = active_trace.get("diagnostics")
        if not diag:
            st.warning("No diagnostic payload attached to this turn.")
            return

        _render_turn_panels(diag, active_trace)


def _render_turn_panels(diagnostics: Mapping[str, Any], trace_meta: Mapping[str, Any]) -> None:
    """Render the 8 independent expandable panels for the selected turn."""
    funnel = diagnostics.get("funnel", {})
    p1 = diagnostics.get("panel_query_understanding", {})
    p2 = diagnostics.get("panel_dense_retrieval", {})
    p3 = diagnostics.get("panel_bm25_retrieval", {})
    p4 = diagnostics.get("panel_hybrid_fusion", {})
    p5 = diagnostics.get("panel_ranking_verification", {})
    p6 = diagnostics.get("panel_packaged_context", {})
    p7 = diagnostics.get("panel_answer_generation", {})
    p8 = diagnostics.get("panel_stage_timings", {})

    # Funnel Metric Ribbon
    st.markdown("---")
    st.markdown(f"##### 📊 Turn #{trace_meta.get('turn_number', 1)} Pipeline Funnel")
    f_cols = st.columns(5)
    f_cols[0].metric("1. Fused Total", funnel.get("retrieved_total", 0))
    f_cols[1].metric("2. Verified", funnel.get("verified_count", 0))
    f_cols[2].metric("3. Rejected", funnel.get("rejected_count", 0))
    f_cols[3].metric("4. Packaged Context", f"{p6.get('context_units_count', 0)} units")
    f_cols[4].metric("5. Displayed Citations", funnel.get("displayed_citations_count", 0))

    st.caption(
        "Progression: **Dense + BM25** → **Weighted RRF** → **Candidate Verification** → "
        "**Context Packaging** → **Answer LLM**"
    )

    # -------------------------------------------------------------
    # Section 1: Request Summary and Terminal Status
    # -------------------------------------------------------------
    with st.expander("1. 📋 Request Summary and Terminal Status", expanded=True):
        s1_c1, s1_c2 = st.columns(2)
        with s1_c1:
            st.markdown(f"**Turn ID:** `{trace_meta.get('turn_id', '')}`")
            st.markdown(f"**Turn Number:** `#{trace_meta.get('turn_number', 1)}`")
            st.markdown(f"**Original User Query:**\n> {trace_meta.get('query', '')}")
            resolved_q = diagnostics.get('resolved_question') or trace_meta.get('query', '')
            st.markdown(f"**Resolved Query:**\n> {resolved_q}")
        with s1_c2:
            term_status = str(trace_meta.get('terminal_status', diagnostics.get('terminal_status', 'success'))).upper()
            status_badge = "🟢" if term_status == "SUCCESS" else ("🟡" if term_status == "SAFE_REFUSAL" else "🔴")
            st.markdown(f"**Terminal Status:** {status_badge} `{term_status}`")
            exec_route = diagnostics.get('execution_route') or diagnostics.get('route_selected', 'rag_hybrid')
            route_label = "⚡ Direct Campus Navigation (Fast Path)" if "navigation" in str(exec_route).lower() else "🔄 Canonical Hybrid RAG Pipeline"
            st.markdown(f"**Execution Route:** `{route_label}`")
            st.markdown(f"**Total Turn Latency:** `{trace_meta.get('latency_seconds', 0.0):.4f}s`")
            st.markdown(f"**Session ID:** `{trace_meta.get('session_id', 'default_session')}`")
            st.markdown(f"**Timestamp:** `{trace_meta.get('timestamp', '')}`")

    # -------------------------------------------------------------
    # Section 2: Query Understanding and Conversation Resolution
    # -------------------------------------------------------------
    with st.expander("2. 🧠 Query Understanding and Conversation Resolution", expanded=False):
        u_in = p1.get("user_input", {})
        interp = p1.get("interpretation", {})
        r_sel = p1.get("route_selection", {})
        handoff = p1.get("retrieval_handoff", {})

        # Route Badge and Overview Ribbon
        sel_route = r_sel.get("selected_route", "A")
        init_route = r_sel.get("initial_route", sel_route)
        llm_calls = r_sel.get("qu_llm_calls", 0)
        route_time = r_sel.get("total_qu_latency_seconds", 0.0)

        route_labels = {
            "A": "⚡ Route A: Deterministic Python Fast Path",
            "B": "⚙️ Route B: Lightweight Hybrid Processing (No LLM)",
            "C": "🧠 Route C: Local LLM Query Planner",
        }
        route_badge = route_labels.get(sel_route, f"Route {sel_route}")

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Selected Route", sel_route, help=route_badge)
        r_col2.metric("QU Wall Time", f"{route_time:.4f}s")
        r_col3.metric("QU LLM Calls", f"{llm_calls}", delta="0 LLM Calls" if llm_calls == 0 else "1 LLM Call", delta_color="normal" if llm_calls == 0 else "inverse")
        r_col4.metric("Escalated", "Yes (A→B/C)" if r_sel.get("escalation_happened") else "No (Direct)")

        st.markdown(f"**Active Execution Path:** `{route_badge}`")
        st.markdown(f"- **Route Choice Reason:** {r_sel.get('route_reason', 'N/A')}")
        if r_sel.get("escalation_happened"):
            st.info(f"🔄 **Safe Escalation Triggered:** Initially routed to `{init_route}`, escalated to `{sel_route}`: {r_sel.get('route_reason')}")
        if llm_calls == 0:
            st.success("✅ **Zero Generative Planning LLMs Invoked**: Query understanding completed entirely on CPU via deterministic Python operations.")
        else:
            st.warning(f"⚠️ **Generative Planning LLM Invoked ({r_sel.get('llm_model', 'qwen2.5:3b')})**: Semantic planning took {r_sel.get('operation_timings', {}).get('query_planning_llm', 0.0):.3f}s.")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### 1. User Input & Conversation Context")
            st.markdown(f"**Exact Original Question:**\n> {u_in.get('original_question', '')}")
            st.markdown(f"**Session ID:** `{u_in.get('session_id', '')}` | **Turn ID:** `{trace_meta.get('turn_id', '')}`")
            history = u_in.get("chat_history", [])
            st.markdown(f"**Chat History Supplied:** {len(history)} messages")
            if history:
                with st.expander("View Prior Context Messages", expanded=False):
                    for h in history:
                        st.caption(f"**{h.get('role', 'user')}:** {h.get('content', '')}")

            st.markdown("###### 2. Sub-Stage Operation Timings")
            op_t = r_sel.get("operation_timings", {})
            st.markdown(f"- **Route Selector:** `{op_t.get('route_selection', 0.0):.5f}s`")
            if op_t.get("route_a_fast_path"):
                st.markdown(f"- **Route A Fast Path:** `{op_t.get('route_a_fast_path', 0.0):.5f}s`")
            if op_t.get("route_b_lightweight"):
                st.markdown(f"- **Route B Lightweight:** `{op_t.get('route_b_lightweight', 0.0):.5f}s`")
            if op_t.get("query_planning_llm"):
                st.markdown(f"- **Planning LLM (qwen2.5:3b):** `{op_t.get('query_planning_llm', 0.0):.4f}s`")
            st.markdown(f"- **Pre-Scope & Guard:** `{op_t.get('guard_and_scope', 0.0):.5f}s`")
            st.markdown(f"- **Total Query Understanding:** `{op_t.get('total_understand_query', route_time):.4f}s`")

        with c2:
            st.markdown("###### 3. Interpretation & Resolved Semantics")
            st.markdown(f"**Resolved Question:**\n> {interp.get('resolved_question', '')}")
            st.markdown(f"**Conversation Mode:** `{interp.get('conversation_mode', 'standalone')}`")
            st.markdown(f"**Active Topic:** `{interp.get('active_topic', 'None')}` | **Active Entity:** `{interp.get('active_entity', 'None')}`")
            st.markdown(f"**Is Multi-Intent:** `{interp.get('is_multi_intent', False)}` | **Query Route:** `{interp.get('query_route', '')}`")
            st.markdown(f"**Scope Decision:** `{interp.get('query_scope', 'in_scope')}` (Confidence: `{interp.get('query_scope_confidence', 1.0)}`)")
            if interp.get("pipeline_stop"):
                st.warning(f"Pipeline Stopped Early: `{interp.get('stop_reason', '')}`")

            st.markdown("###### 4. Detected Entities & Linguistic Facets")
            ents = interp.get("detected_entities", [])
            attrs = interp.get("detected_attributes", [])
            dates = interp.get("detected_dates", [])
            nums = interp.get("detected_numbers", [])
            quals = interp.get("detected_qualifiers", [])

            st.markdown(f"- **Entities:** {', '.join(f'`{e}`' for e in ents) if ents else '*None*'}")
            st.markdown(f"- **Attributes:** {', '.join(f'`{a}`' for a in attrs) if attrs else '*None*'}")
            st.markdown(f"- **Dates / Periods:** {', '.join(f'`{d}`' for d in dates) if dates else '*None*'}")
            st.markdown(f"- **Numbers / Metrics:** {', '.join(f'`{n}`' for n in nums) if nums else '*None*'}")
            st.markdown(f"- **Qualifiers / Constraints:** {', '.join(f'`{q}`' for q in quals) if quals else '*None*'}")

        st.markdown("---")
        st.markdown("###### 5. Retrieval Handoff & Intent Associations")
        h_cols = st.columns(3)
        h_cols[0].markdown(f"**Recipient Node:** `{handoff.get('recipient_node', '')}`")
        h_cols[1].markdown(f"**Collection / Tenant:** `{handoff.get('vector_collection', '')}`")
        h_cols[2].markdown(f"**Contract Status:** `{handoff.get('contract_validation_status', 'VALID')}`")

        sq_with_intents = handoff.get("search_queries_with_intents", [])
        if sq_with_intents:
            st.markdown("**Dispatched Search Queries & Associated Intent Units:**")
            for sq_i, item in enumerate(sq_with_intents, start=1):
                st.markdown(f"- **Query {sq_i}** (weight `{item.get('weight', 1.0)}`): `{item.get('query')}`  \n  ↳ *Associated Intent:* {item.get('associated_intent')}")
        else:
            r_queries = handoff.get("retrieval_queries", [])
            r_weights = handoff.get("retrieval_weights", [])
            for q_idx, rq in enumerate(r_queries):
                w = r_weights[q_idx] if q_idx < len(r_weights) else 1.0
                st.markdown(f"- Query {q_idx + 1} (weight `{w}`): `{rq}`")

        with st.expander("View Exact JSON Payload Passed to Retrieval Node", expanded=False):
            st.json(handoff.get("exact_payload", {}))

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Section 3: Dense, BM25 and Lexical Retrieval
    # -------------------------------------------------------------
    with st.expander("3. 🔍 Dense, BM25 and Lexical Retrieval", expanded=False):
        if p2.get("status") == "intentionally_skipped" or p3.get("status") == "intentionally_skipped":
            st.info(f"⏭️ **Retrieval Intentionally Skipped (Fast Path):** {p2.get('reason') or p3.get('reason') or 'Direct facility match bypassed full vector search.'}")
        dense_candidates = p2.get("candidates", [])
        bm25_candidates = p3.get("candidates", [])

        st.markdown(f"###### 1. Raw Vector Search Hits ({p2.get('candidate_count', 0)} candidates)")
        st.caption("Returned by Chroma embedding similarity (L2 distance: lower distance = closer match).")
        if not dense_candidates:
            st.info("No dense candidates recorded for this turn.")
        else:
            for d in dense_candidates:
                d_rank = d.get("rank", 1)
                d_score = d.get("score")
                score_label = f"L2 Distance: `{d_score:.4f}`" if d_score is not None else "Score: N/A"
                src_short = d.get("source", "").split("/")[-1]
                doc_id = d.get("chunk_id", "")[:12]
                title = f"Dense Rank #{d_rank} • {src_short} (`{doc_id}...`) — {score_label}"
                with st.expander(title, expanded=False):
                    st.markdown(f"**Full Source:** `{d.get('source')}` | **Chunk ID:** `{d.get('chunk_id')}`")
                    if d.get("query"):
                        st.markdown(f"**Retrieved by Query:** `{d.get('query')}`")
                    st.code(d.get("text", "")[:400], language="text")

        st.markdown("---")
        st.markdown(f"###### 2. Raw BM25 Keyword Hits ({p3.get('candidate_count', 0)} candidates)")
        st.caption("Returned by BM25Okapi inverted index matching exact keywords, codes, and tokens.")
        if not bm25_candidates:
            st.info("No BM25 candidates recorded for this turn.")
        else:
            for b in bm25_candidates:
                b_rank = b.get("rank", 1)
                b_score = b.get("score")
                score_label = f"BM25 Score: `{b_score:.4f}`" if b_score is not None else "Score: N/A"
                src_short = b.get("source", "").split("/")[-1]
                doc_id = b.get("chunk_id", "")[:12]
                title = f"BM25 Rank #{b_rank} • {src_short} (`{doc_id}...`) — {score_label}"
                with st.expander(title, expanded=False):
                    st.markdown(f"**Full Source:** `{b.get('source')}` | **Chunk ID:** `{b.get('chunk_id')}`")
                    if b.get("query"):
                        st.markdown(f"**Retrieved by Query:** `{b.get('query')}`")
                    st.code(b.get("text", "")[:400], language="text")

    # -------------------------------------------------------------
    # Section 4: Hybrid Fusion Results
    # -------------------------------------------------------------
    with st.expander("4. 🔀 Hybrid Fusion Results (Top 15 Weighted RRF)", expanded=False):
        if p4.get("status") == "intentionally_skipped":
            st.info(f"⏭️ **Hybrid Fusion Intentionally Skipped (Fast Path):** {p4.get('reason', 'Direct facility match bypassed candidate fusion.')}")
        top15 = p4.get("top_15_candidates", [])
        st.markdown(f"###### Reciprocal Rank Fusion Output (Top {len(top15)} of {p4.get('total_fused_count', 0)} fused)")
        st.caption("Fused across Dense, BM25, and Lexical channels using Weighted RRF (k=60).")

        if not top15:
            st.info("No fused candidates available.")
        else:
            for cand in top15:
                init_rank = cand.get("initial_rank", 1)
                final_rank = cand.get("final_rank")
                status = cand.get("verification_status", "unverified")
                status_icon = "✅" if status == "verified" else ("⚠️" if status == "uncertain" else "❌")
                src_short = cand.get("source", "").split("/")[-1]
                doc_id = cand.get("document_id", "")[:12]
                rrf_score = cand.get("fusion_score", 0.0)

                label = f"Fused #{init_rank} {status_icon} — {src_short} (`{doc_id}...`) — RRF: `{rrf_score:.6f}`"
                with st.expander(label, expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Source:** `{cand.get('source')}`")
                        st.markdown(f"**Department:** {cand.get('department')} | **Prog:** {cand.get('programme')}")
                        st.markdown(f"**Doc Type:** `{cand.get('doc_type')}`")
                        st.markdown(f"**In Final LLM Context:** {'✅ Yes' if cand.get('in_final_context') else '❌ No'}")
                    with c2:
                        d_r = cand.get("dense_rank")
                        d_s = cand.get("dense_score")
                        b_r = cand.get("bm25_rank")
                        b_s = cand.get("bm25_score")
                        st.markdown(f"**Dense Rank / L2 Distance:** `{d_r}` / `{d_s}`")
                        st.markdown(f"**BM25 Rank / BM25 Score:** `{b_r}` / `{b_s}`")
                        st.markdown(f"**Verification Outcome:** `{status.upper()}` (Score: `{cand.get('verification_score', 0.0)}`)")
                        if cand.get("verification_reasons"):
                            st.markdown(f"**Reasons:** {', '.join(cand.get('verification_reasons', []))}")

                    st.markdown("**Candidate Excerpt:**")
                    st.code(cand.get("text_snippet", ""), language="text")

    # -------------------------------------------------------------
    # Section 5: Verification and Ranking Decisions
    # -------------------------------------------------------------
    with st.expander("5. ⚖️ Verification and Ranking Decisions", expanded=False):
        if p5.get("status") == "intentionally_skipped":
            st.info(f"⏭️ **Verification Intentionally Skipped (Fast Path):** {p5.get('reason', 'Structured institutional facility directory provides authenticated facts.')}")
        decisions = p5.get("decisions", [])
        st.markdown("###### Verification & Reranking Outcomes")
        st.caption(
            "Verifies candidate relevance against the active question before admission into the final LLM context. "
            "Eliminates false rejections and logs explicit decision reasons."
        )

        v_cols = st.columns(4)
        v_cols[0].metric("Verified", p5.get("verified_count", 0))
        v_cols[1].metric("Uncertain", p5.get("uncertain_count", 0))
        v_cols[2].metric("Rejected", p5.get("rejected_count", 0))
        v_cols[3].metric("Ranked for Context", p5.get("ranked_count", 0))

        if not decisions:
            st.info("No decision records available.")
        else:
            for dec in decisions:
                stat = dec.get("verification_status", "unknown")
                icon = "✅" if stat == "verified" else ("⚠️" if stat == "uncertain" else "❌")
                i_rank = dec.get("initial_rank")
                f_rank = dec.get("final_rank")
                rank_str = f"Final #{f_rank}" if f_rank else "Filtered/Unranked"
                src_short = dec.get("source", "").split("/")[-1]
                reasons = ", ".join(dec.get("reasons", [])) or "None recorded"

                st.markdown(
                    f"{icon} **Fused #{i_rank} → {rank_str}** | `{src_short}` | Status: **{stat.upper()}** | Stage: `{dec.get('stage')}`"
                )
                if reasons != "None recorded":
                    st.caption(f"Decision Reasons: *{reasons}*")
                st.markdown("---")

    # -------------------------------------------------------------
    # Section 6: Final Packaged Context and Token Budget
    # -------------------------------------------------------------
    with st.expander("6. 📦 Final Packaged Context and Token Budget", expanded=False):
        if p6.get("status") == "intentionally_skipped":
            st.info("⏭️ **Context Packaging Intentionally Skipped (Fast Path):** Direct facility card rendered without document context packaging.")
        units = p6.get("packaged_units", [])
        tokens = p6.get("estimated_tokens", 0)
        budget = p6.get("token_budget", 4000)
        pct = p6.get("token_budget_percentage", 0.0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Packaged Context Units", len(units))
        c2.metric("Estimated Context Tokens", f"{tokens} / {budget}")
        c3.metric("Budget Utilization", f"{pct}%")
        st.progress(min(1.0, float(pct) / 100.0))

        st.markdown(f"###### Ordered Evidence Units Passed to LLM ({len(units)} units):")
        if not units:
            st.info("No evidence units packaged (direct response or safe refusal).")
        else:
            for idx, u in enumerate(units, start=1):
                u_src = u.get("source", "").split("/")[-1]
                with st.expander(f"Unit #{idx}: {u_src} — {u.get('heading', '')}", expanded=False):
                    st.caption(f"Evidence ID: `{u.get('evidence_id')}` | Source Path: `{u.get('source')}`")
                    st.code(u.get("text", ""), language="text")

        raw_ctx = p6.get("raw_context_text", "")
        if raw_ctx:
            with st.expander("View Complete Raw Assembled Context String", expanded=False):
                st.code(raw_ctx, language="text")

        prompt_msgs = p6.get("prompt_messages", [])
        if prompt_msgs:
            with st.expander("View Exact Formatted Prompt Messages Passed to Answer Model", expanded=False):
                for m_idx, pm in enumerate(prompt_msgs):
                    st.markdown(f"**Message {m_idx + 1} ({pm.get('role', 'unknown')}):**")
                    st.code(pm.get("content", ""), language="text")

    # -------------------------------------------------------------
    # Section 7: Answer Generation, Validation and Citations
    # -------------------------------------------------------------
    with st.expander("7. ✍️ Answer Generation, Validation and Citations", expanded=False):
        if p7.get("status") == "direct_structured_render":
            st.success("⚡ **Direct Structured Handler:** Rendered authentic facility card directly from institutional registry (0 LLM model calls).")
        ans = p7.get("answer", "")
        cits = p7.get("supporting_citations", [])
        g_stat = p7.get("grounding_status", "not_run")
        guard_stat = p7.get("guard_status", "ok")

        st.markdown(f"**Answer Mode:** `{p7.get('answer_mode', 'model')}` | **LLM Model Calls:** `{p7.get('model_calls', 0)}`")
        st.markdown(f"**Grounding Status:** `{g_stat}` | **Guardrail Status:** `{guard_stat}`")
        if p7.get("guard_reason"):
            st.warning(f"Guardrail Note: {p7.get('guard_reason')}")

        st.markdown("###### Final Answer Emitted:")
        st.markdown(ans if ans else "*(Empty answer)*")

        st.markdown("###### Displayed Supporting Citations:")
        if not cits:
            st.info("No citations displayed (Safe abstention or ungrounded).")
        else:
            for c_i, c in enumerate(cits, start=1):
                c_src = c.get("source", "").split("/")[-1]
                st.markdown(f"**Citation {c_i}:** `{c_src}` — *{c.get('heading', '')}*")
                st.caption(f"Path: `{c.get('source')}`")
                st.info(c.get("text", ""))

    # -------------------------------------------------------------
    # Section 8: Stage Timings, Reconciliation, Warnings and Errors
    # -------------------------------------------------------------
    with st.expander("8. ⏱️ Stage Timings, Reconciliation, Warnings and Errors", expanded=False):
        timings = p8.get("timings", {})
        warnings = p8.get("warnings", [])
        errors = p8.get("errors", [])

        st.markdown("###### Pipeline Stage Latencies:")
        if not timings:
            st.info("No individual stage timing metrics recorded.")
        else:
            total_lat = (
                trace_meta.get("latency_seconds")
                or timings.get("total_turn")
                or timings.get("total_pipeline")
                or (
                    sum(
                        v for k, v in timings.items()
                        if isinstance(v, (int, float)) and k in ("understand_query", "dense_retrieve", "verify_and_rank", "context_packaging", "llm_generation")
                    )
                )
            )
            t_rows = [
                ("Conversation Resolution", timings.get("resolve_conversation")),
                ("Query Understanding", timings.get("understand_query")),
                ("Planning LLM", timings.get("query_planning_llm")),
                ("Query Embedding (nested in Dense)", timings.get("query_embedding")),
                ("Chroma Search (nested in Dense)", timings.get("chroma_vector_search")),
                ("Dense Retrieval (Total)", timings.get("dense_retrieve")),
                ("BM25 Retrieval", timings.get("bm25_retrieve")),
                ("Deterministic Lexical Scan", timings.get("lexical_retrieve")),
                ("Hybrid Fusion (RRF)", timings.get("fusion")),
                ("Verification & Ranking", timings.get("verify_and_rank")),
                ("Context Packaging", timings.get("context_packaging")),
                ("Answer LLM Generation", timings.get("llm_generation")),
                ("Post-Processing & Citations", timings.get("post_processing")),
                ("Total Pipeline Latency", total_lat),
            ]
            t_cols = st.columns(2)
            for idx, (stage_name, val) in enumerate(t_rows):
                col = t_cols[idx % 2]
                val_str = f"`{val:.3f}s`" if val is not None and isinstance(val, (int, float)) and val > 0 else ("*Skipped/None*" if val is None else f"`{val:.3f}s`")
                col.markdown(f"- **{stage_name}:** {val_str}")

            exclusive = p8.get("exclusive_timings", {})
            if exclusive and isinstance(exclusive, dict):
                excl_sum = sum(v for k, v in exclusive.items() if isinstance(v, (int, float)))
                diff = abs(float(total_lat or 0.0) - excl_sum)
                st.caption(
                    f"ℹ️ **Timing Contract:** *Exclusive stage sum: `{excl_sum:.3f}s` | "
                    f"Total pipeline: `{total_lat:.3f}s` (Reconciliation delta: `{diff:.3f}s` framework/graph dispatch)*"
                )

        if warnings:
            st.markdown("###### ⚠️ Runtime Warnings:")
            for w in warnings:
                st.warning(str(w))

        if errors:
            st.markdown("###### ❌ Runtime Errors:")
            for e in errors:
                st.error(str(e))


def render_retrieval_diagnostics_panel(diagnostics: Mapping[str, Any] | None) -> None:
    """Render the inline developer retrieval diagnostic score box (backward compatibility)."""
    if not diagnostics or not isinstance(diagnostics, Mapping):
        return

    real_lat = float(
        diagnostics.get("latency_seconds")
        or (
            diagnostics.get("panel_stage_timings", {}).get("timings", {}).get("total_turn")
            if isinstance(diagnostics.get("panel_stage_timings"), Mapping)
            else 0.0
        )
        or 0.0
    )

    # Delegate to the unified panel renderer using real trace meta
    trace_meta = {
        "turn_id": str(diagnostics.get("turn_id", "active_turn")),
        "turn_number": diagnostics.get("turn_number", 1),
        "timestamp": str(diagnostics.get("timestamp", "")),
        "query_preview": str(diagnostics.get("user_query", "Current Query"))[:45],
        "latency_seconds": real_lat,
    }

    with st.expander("🛠️ Retrieval Diagnostics & Evidence Inspector", expanded=False):
        _render_turn_panels(diagnostics, trace_meta)



__all__ = [
    "render_developer_trace_inspector",
    "render_retrieval_diagnostics_panel",
]
