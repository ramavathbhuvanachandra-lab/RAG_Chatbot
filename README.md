IIT Jodhpur AI Assistant --- RAG Chatbot

Current Version: 3.3
Application: Streamlit
Primary Purpose: Conversational, evidence-grounded institutional AI
assistant for IIT Jodhpur

Overview

The IIT Jodhpur AI Assistant is a Retrieval-Augmented Generation (RAG)
application designed to answer institute-level questions using a curated
institutional knowledge base.

The system is designed around one central principle:

The assistant should be grounded in a reviewable institutional
knowledge base, not behave like a general-purpose chatbot that invents
institutional facts.

The project evolved through multiple engineering phases covering website
knowledge acquisition, knowledge preservation and organization,
retrieval, conversational reasoning, evidence validation, answer
grounding, regression testing, and production-oriented hardening.

The current repository contains the V3.3 chatbot application and its
supporting RAG, conversation, evidence, database, UI, evaluation, and
regression-testing components.

Product Positioning

This project is not intended to be "just another FAQ chatbot."

The broader product direction is:

Conversational Institutional AI for Higher Education

The assistant is intended to help students, faculty, staff, applicants,
visitors, and other users find reliable information about IIT Jodhpur.

Typical areas include:

Admissions

Academic programs

Academic regulations

Registration and academic administration

Departments and schools

Faculty and institutional people

Research

Laboratories and workshops

Campus and infrastructure

Hostels and student residential life

Finance and fees

Health, safety, and wellbeing

Student services

General institute information

The assistant should answer only to the extent supported by the
available institutional evidence.

System Architecture

The project has two closely related knowledge/RAG layers:

Knowledge preparation / ingestion

Runtime conversational RAG chatbot

The overall lifecycle is:

                    IIT JODHPUR WEBSITE / SOURCES
                               │
                               ▼
                     WEBSITE KNOWLEDGE ACQUISITION
                               │
                               ▼
                    RAW / SOURCE-OF-TRUTH DATA
                               │
                               ▼
                    KNOWLEDGE STRUCTURING
                               │
                               ▼
                    KNOWLEDGE ORGANIZATION
                               │
                               ▼
                    SEMANTIC RAG PLANNING
                               │
                               ▼
                     FINAL RAG DOCUMENTS
                               │
                               ▼
                         INGESTION
                               │
                               ▼
                    DOCUMENT CHUNKING
                               │
                               ▼
                 EMBEDDINGS + VECTOR STORE
                               │
                               ├───────────────┐
                               ▼               ▼
                        DENSE RETRIEVAL     BM25
                               │               │
                               └───────┬───────┘
                                       ▼
                              WEIGHTED RRF FUSION
                                       │
                                       ▼
                              SCOPE / RELEVANCE
                                       │
                                       ▼
                         DUPLICATE / DIVERSITY CONTROL
                                       │
                                       ▼
                            LOCAL CONTEXT EXPANSION
                                       │
                                       ▼
                             EVIDENCE GROUPING
                                       │
                                       ▼
                         EVIDENCE SUFFICIENCY
                                       │
                                       ▼
                           EVIDENCE COVERAGE
                                       │
                              ┌────────┴────────┐
                              │                 │
                           ENOUGH          INSUFFICIENT
                              │                 │
                              ▼                 ▼
                        GROUNDED ANSWER      FALLBACK
                              │
                              ▼
                        ANSWER GUARD
                              │
                              ▼
                         STREAMLIT UI

Runtime Conversation Flow

For a user question, the production LangGraph workflow follows a
controlled sequence.

User Question
     │
     ▼
Conversation Resolution
     │
     ▼
Multi-Intent Planning
     │
     ├──────────── Single Intent ────────────┐
     │                                      │
     └──────────── Multi Intent              │
                    │                        │
                    ▼                        │
             Independent Evidence            │
               Pipeline per Intent           │
                    │                        │
                    └──────────────┬─────────┘
                                   ▼
                         Dense + BM25 Retrieval
                                   │
                                   ▼
                           Weighted RRF Fusion
                                   │
                                   ▼
                              Deduplication
                                   │
                                   ▼
                           Initial Reranking
                                   │
                                   ▼
                        Local Context Expansion
                                   │
                                   ▼
                           Evidence Grouping
                                   │
                                   ▼
                           Group Ranking
                                   │
                                   ▼
                        Evidence Sufficiency
                                   │
                                   ▼
                         Evidence Coverage
                                   │
                                   ▼
                           Final Evidence
                                   │
                                   ▼
                         Context Compression
                                   │
                                   ▼
                         One Answer LLM Call
                                   │
                                   ▼
                             Answer Guard
                                   │
                                   ▼
                              Final Answer

The multi-intent design deliberately avoids making a separate
answer-generation LLM call for every intent. Each intent receives an
independent evidence pipeline, and the final response is composed once.

Core Design Principles

1. Evidence first

Institutional facts should come from retrieved institutional knowledge.

The answer model should not fill missing information using general world
knowledge.

2. No unsupported invention

The assistant should not invent:

fees

eligibility rules

course codes

department information

contact paths

URLs

policies

dates

institutional procedures

If the available evidence does not support an answer, the system should
fall back rather than hallucinate.

3. Conversation history is contextual, not factual

Conversation history is used to understand references such as:

"What about the fees?"

after:

"Tell me about hostel facilities."

History helps resolve what "fees" refers to, but factual claims still
need retrieval evidence.

4. Scope preservation

The system protects important boundaries such as:

B.Tech vs M.Tech vs M.Sc. vs Ph.D.

department vs department

program vs program

academic topic vs unrelated topic

specific entity vs broad institutional topic

A retrieved document should not become evidence for a different scope
merely because it contains similar words.

5. Broad questions should remain broad

A broad question should not be accidentally narrowed to whichever
document happened to rank first.

For example, a question asking about IIT Jodhpur's programs should not
be answered using only one program unless the evidence itself is limited
to that scope.

6. Generic mechanisms over hardcoded fixes

The project avoids fixes such as:

"If the question contains B5, return document X."

Instead, the goal is to improve generic retrieval, scope, evidence, and
conversation mechanisms so the solution works across the institution.

7. Minimal, maintainable engineering

The project follows:

smallest safe change

explicit interfaces

separation of responsibilities

regression testing

evidence-driven debugging

no unnecessary architecture rewrites

no premature optimization

Retrieval Architecture

The retrieval layer combines semantic and lexical retrieval.

Dense retrieval

Vector retrieval captures semantic similarity between the user query and
knowledge chunks.

BM25 retrieval

BM25 provides lexical/keyword recall and is particularly useful for:

names

course codes

identifiers

exact terminology

uncommon institutional phrases

Weighted Reciprocal Rank Fusion

The two retrieval signals are combined using weighted RRF.

Current retrieval design keeps:

Dense retrieval : 0.7
BM25            : 0.3

The system then performs additional relevance, scope, deduplication, and
evidence processing before final answer generation.

The retrieval layer also supports bounded query variants rather than
allowing uncontrolled query expansion.

Query Processing

The chatbot supports query rewriting and bounded multi-query retrieval.

The query planner is designed to preserve:

original meaning

original scope

exact identifiers

course/program terminology

department/entity references

Generated queries must not silently change a broad question into a
narrow one or a specific question into a broader one.

For example:

Original:
What is the timetable for Group B5?

Allowed:
What is the class schedule for Group B5 at IIT Jodhpur?
Where can I find the timetable or schedule for Group B5?

Not allowed:
What are the courses in Section B?
What is the timetable for Group B4?

Evidence and Grounding

A major part of the later development focused on moving beyond simple
"retrieve top-k and ask the LLM" behavior.

The runtime contains dedicated components for:

evidence extraction

evidence grouping

evidence coverage

evidence sufficiency

claim-scope checking

final evidence-scope filtering

temporal/numeric evidence handling

claim-context filtering

contradiction handling

supported-claim preservation

answer grounding

answer guarding

The objective is to ensure that the final answer is supported by the
evidence that was actually retrieved.

Conversational RAG

The chatbot supports multi-turn interaction.

Conversation handling includes:

conversation/session state

follow-up question resolution

entity/topic continuity

natural topic switching

multi-intent questions

context-aware query processing

A short question such as:

What about fees?

can be interpreted using recent conversation context when the reference
is sufficiently clear.

The conversation layer should not manufacture factual information that
is absent from the knowledge base.

Knowledge Ingestion and RAG Data

The chatbot consumes prepared institutional knowledge rather than
directly treating arbitrary web pages as answer context.

The upstream knowledge workflow evolved toward semantic organization:

Raw Website Content
       │
       ▼
Structured Knowledge
       │
       ▼
Organized Knowledge
       │
       ▼
Semantic RAG Plan
       │
       ▼
Human-readable RAG DOCX
       │
       ▼
Chatbot Ingestion

The organization strategy is semantic/domain based rather than simply
copying website URL structure.

Examples of logical knowledge areas include:

IIT Jodhpur Knowledge
├── Academics
│   ├── Programs
│   ├── Regulations
│   ├── Timetables
│   ├── Minors
│   └── Fees
│
├── Departments and Schools
│   ├── Overview
│   ├── Programs
│   ├── Faculty
│   ├── Research
│   ├── Labs
│   └── Projects
│
├── Admissions
├── Campus
├── Student Life
├── Hostels
├── Research
├── Finance
└── Institutional Services

The exact number of final documents is intentionally not treated as a
fixed architectural requirement. The goal is:

maximum semantic coherence

minimum fragmentation

minimum unnecessary duplication

zero unjustified information loss

Development Journey

The system was developed incrementally rather than rewritten from
scratch.

The major hardening sequence was:

Phase 1
Crawler
   ↓
Phase 2
Knowledge Preservation / Structuring
   ↓
Phase 3
Knowledge Organization
   ↓
Phase 4
RAG Retrieval
   ↓
Phase 5
Conversational RAG / Prompts
   ↓
Phase 6
LLM / Latency
   ↓
Phase 7
Regression Testing
   ↓
Phase 8-series
Broad Institutional / Retrieval Hardening
   ↓
Phase 8.3
Knowledge Organization
   ↓
Phase 8.4
Semantic RAG Planning
   ↓
Phase 8.5
Final RAG DOCX Building + Validation

The project reached the current V3.3 application checkpoint after
this iterative development and hardening work.

Knowledge Pipeline Milestones

The upstream ingestion work established several important milestones.

Website crawling

The crawler was developed to discover institutional pages while
preserving useful source relationships.

Page extraction

The extraction layer preserves main content and breadcrumb information.

Navigation understanding

Navigation processing identifies candidate navigation structures,
analyzes redundancy, and builds navigation trees.

The navigation model is intentionally simple:

NavigationNode
├── text
├── href
└── children

Knowledge structuring

Structured knowledge preserves the useful information from the source
while separating it from raw website presentation.

Knowledge organization

The organizer groups knowledge into semantically meaningful domains
rather than blindly following website paths.

Semantic RAG planning

The RAG planner determines suitable document groupings from the
available knowledge rather than requiring every institution to use the
same hardcoded document taxonomy.

Final RAG document generation

The final RAG builder creates human-readable DOCX documents from the
plan.

A key invariant is:

Planned Knowledge Units
        =
Final RAG Knowledge Units

Useful knowledge should not silently disappear during final document
generation.

Phase 8.5 Validation

The final RAG document builder validates:

planned documents

generated documents

input knowledge units

output knowledge units

missing units

extra units

duplicate units

coverage completeness

The intended result is:

Input Units == Output Units
Missing Units == 0
Extra Units == 0
Duplicate Output Units == 0
Coverage == PASS

This is important because a beautifully formatted RAG document is not
useful if information was silently lost during transformation.

Testing and Evaluation

Testing is treated as a first-class part of the system.

The repository contains tests covering several layers.

Answer tests

Examples include:

answer claim auditing

answer grounding

scope control

supported claim preservation

multi-intent answer behavior

Conversation tests

Examples include:

conversation resolver

follow-up grounding

multi-turn continuity

topic switching

session behavior

Retrieval tests

Examples include:

dense retrieval

BM25 retrieval

RRF

reranking

scope protection

evidence groups

evidence sufficiency

evidence coverage

local context

retrieval diversity

multi-intent retrieval

Database tests

Examples include:

persistence

authentication

Supabase connectivity/failure handling

session/database boundaries

Phase 8 regression tests

The Phase 8 test suite contains broad institutional retrieval and
stability checks.

Representative regression areas include:

exact identifier retrieval

wrong-document retrieval

cross-department contamination

cross-topic contamination

query-rewrite errors

multi-query drift

follow-up context

timetable retrieval

table interpretation

grounding

hallucination resistance

numerical accuracy

course-code accuracy

broad vs specific questions

out-of-scope questions

prompt injection

session/history behavior

UI behavior

latency

Docker/deployment

metadata correctness

duplicates/conflicts

Important Historical Canary Cases

Several real-world cases were used to expose weaknesses in the system.

Important examples include:

B5 timetable

Minor Programs

M.Tech

Academic Regulations

Electrical Engineering faculty

Electrical Engineering research

These cases were useful because they exposed different classes of
failure:

missing evidence

incorrect interpretation of tables

cross-department contamination

incorrect scope

broad-query narrowing

incomplete answers

follow-up context loss

The objective was not to hardcode solutions for these individual
questions, but to use them as regression cases for generic mechanisms.

Repository Structure

The repository is organized into application, backend, UI, data,
evaluation, diagnostics, and testing layers.

RAG_Chatbot/
│
├── app.py
├── ingest.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements_gpu.txt
├── README.md
│
├── backend/
│   ├── chatbot.py
│   ├── graph.py
│   ├── nodes.py
│   ├── state.py
│   ├── retriever.py
│   ├── vectorstore.py
│   ├── embedding.py
│   ├── llm.py
│   ├── prompts.py
│   │
│   ├── conversation_resolver.py
│   ├── conversation_session.py
│   ├── multi_intent.py
│   ├── multi_intent_answer.py
│   │
│   ├── evidence.py
│   ├── evidence_groups.py
│   ├── evidence_coverage.py
│   ├── evidence_scope_gate.py
│   ├── final_evidence_scope.py
│   ├── claim_scope.py
│   ├── claim_context_filter.py
│   ├── answer_grounding.py
│   ├── answer_guard.py
│   │
│   ├── retrieval_diversity.py
│   ├── retrieval_query_planner.py
│   ├── local_context.py
│   └── ...
│
├── UI/
│   ├── chat.py
│   ├── components.py
│   ├── dashboard.py
│   ├── login.py
│   ├── sidebar.py
│   ├── campus_map.py
│   └── styles.py
│
├── data/
│   ├── campus_locations.json
│   ├── emergency_contacts.json
│   └── data_iitj/
│       └── iitj_rag_v1_docs_production/
│
├── evaluation/
│   ├── dataset_v1.csv
│   ├── evaluators.py
│   ├── langsmith_eval.py
│   ├── prompts.py
│   └── run_eval.py
│
├── diagnostics/
│   └── retrieval/
│
├── migrations/
│   └── 001_normalize_database_identity.sql
│
├── scripts/
│
└── tests/
    ├── answer/
    ├── conversation/
    ├── database/
    ├── evaluation/
    ├── phase5/
    ├── phase7/
    ├── phase8/
    ├── retrieval/
    └── ...

Generated/runtime artifacts such as vector databases, logs, reports,
caches, and temporary files are intentionally excluded from source
control.

Main Components

app.py

Main Streamlit application entry point.

UI/

Contains the presentation layer:

chat interface

sidebar

login

dashboard

campus map

reusable components

styling

backend/graph.py

Defines the LangGraph workflow and routing between the single-intent and
multi-intent paths.

backend/nodes.py

Contains the major production workflow nodes for:

conversation resolution

multi-intent planning

retrieval

fusion

reranking

context expansion

evidence assessment

context compression

answer generation

backend/retriever.py

Implements the retrieval layer, including dense retrieval, BM25, RRF,
deduplication, and relevance handling.

backend/vectorstore.py

Provides access to the vector database.

backend/embedding.py

Handles embedding configuration and document representation.

backend/llm.py

Provides the language-model interface used by the application.

backend/prompts.py

Contains prompts governing query/answer behavior and grounding rules.

backend/conversation_resolver.py

Resolves follow-up questions and conversational references while
preserving the active topic/entity.

Evidence modules

The evidence-related modules provide the safeguards between retrieval
and answer generation.

ingest.py

Entry point for loading the prepared knowledge corpus into the runtime
RAG system.

Data Flow at Runtime

The prepared IIT Jodhpur documents are loaded from:

data/data_iitj/

The runtime ingestion layer loads documents, splits them into chunks,
and makes them available to the retrieval system.

The vector database is runtime-generated and is not intended to be
treated as source data.

Conceptually:

Prepared DOCX
     ↓
Document Loader
     ↓
Chunking
     ↓
Embeddings
     ↓
Chroma / Vector Store
     │
     └──────────────► Dense Retrieval
                         │
BM25 Index ──────────────┤
                         ▼
                    RRF Fusion
                         ▼
                 Evidence Pipeline
                         ▼
                    Answer Model

Running the Application

1. Create/activate a Python environment

The project uses Python and Streamlit.

A project-specific environment is recommended rather than using an
unrelated global environment.

Example with Conda:

conda create -n iitj_rag python=3.11 -y
conda activate iitj_rag

Verify:

python --version
which python

2. Install dependencies

python -m pip install -r requirements.txt

For environments that specifically require the GPU dependency set:

python -m pip install -r requirements_gpu.txt

Do not install both requirement files blindly. Use the dependency set
appropriate for the target environment.

3. Run Streamlit

Prefer:

python -m streamlit run app.py

The application uses Streamlit's default local web interface unless
deployment configuration changes the address/port.

Docker

The repository includes a Dockerfile based on Python 3.11.

The container:

creates /app

installs requirements.txt

copies the project

exposes port 8501

launches Streamlit

Build:

docker build -t iitj-rag-chatbot .

Run:

docker run --rm -p 8501:8501 iitj-rag-chatbot

The Docker image should receive any required secrets/configuration
through the deployment environment rather than committing them to the
repository.

Configuration and Secrets

Sensitive configuration should be supplied through environment variables
or the deployment environment.

Do not commit:

API keys

database credentials

tokens

private keys

local secrets

personal environment configuration

The repository .gitignore is configured to exclude common secret,
cache, runtime, log, and temporary files.

Rebuilding the RAG Store

The prepared knowledge documents live under the repository's data
directory.

The general process is:

Update prepared knowledge
        ↓
Run ingestion
        ↓
Rebuild runtime vector store
        ↓
Run retrieval/regression checks
        ↓
Run chatbot

The generated vector database is a runtime artifact and can be recreated
from the prepared knowledge corpus.

Evaluation Workflow

Evaluation is separated from the production chatbot path.

The repository contains:

evaluation/

for evaluation datasets, evaluators, prompts, and evaluation runners.

The broader testing strategy is:

Unit Tests
    ↓
Integration Tests
    ↓
Retrieval Diagnostics
    ↓
Regression Questions
    ↓
Broad Institutional Evaluation
    ↓
Real-world Validation

A change should not be considered successful merely because one question
starts working.

The preferred validation pattern is:

Identify failure
      ↓
Locate failing layer
      ↓
Fix generic mechanism
      ↓
Add regression test
      ↓
Run existing regression suite
      ↓
Run representative real-world cases
      ↓
Compare results

Quality Targets

The project established engineering targets for future production
hardening.

These are targets, not claims of current measured performance:

Metric                                Target

Known-answer accuracy                  ≥ 90%
Conversational follow-up accuracy      ≥ 90%
Unknown/refusal correctness            ≥ 95%
Wrong-scope answers                    < 5%
Internal leakage                           0
Invented contact fallback                  0
Runtime failures                           0

These targets should only be reported as achieved when supported by
repeatable regression/evaluation evidence.

Known Engineering Concerns

The project history identified several classes of problems that
motivated the hardening work:

incorrect document retrieval

cross-department contamination

cross-topic contamination

broad-query narrowing

query-rewrite drift

multi-query drift

table interpretation problems

missing follow-up context

insufficient evidence

hallucination risk

numerical accuracy

course-code accuracy

official-link preservation

database/persistence failures

latency

deployment robustness

The system architecture contains dedicated mechanisms and regression
tests for many of these areas.

Future changes should continue to diagnose the failing layer before
changing the LLM, retriever, data, or prompts.

Engineering Workflow

The preferred development loop is:

1. Reproduce
      ↓
2. Inspect evidence
      ↓
3. Identify the failing layer
      ↓
4. Make the smallest generic change
      ↓
5. Add/update regression test
      ↓
6. Run targeted test
      ↓
7. Run broader regression tests
      ↓
8. Review Git diff
      ↓
9. Commit with a clear message
      ↓
10. Verify clean working tree

Avoid:

fixing individual URLs with hardcoded rules

changing multiple architectural layers simultaneously

deleting data to hide retrieval problems

increasing Top-K without understanding the failure

changing the LLM before proving the LLM is the failing layer

rewriting working components unnecessarily

Source-of-Truth Principle

The institution is the ultimate authority.

The system is designed so that institutional knowledge can eventually
be:

reviewed

corrected

added

removed

approved

The objective is therefore not:

"The chatbot knows everything."

The objective is:

"The assistant is grounded in a reviewable institutional knowledge
base."

Current Repository Status

Application version: V3.3

The repository currently represents a cleaned and organized chatbot
codebase containing:

Streamlit application

UI components

LangGraph orchestration

hybrid retrieval

vector retrieval

BM25

RRF

query planning

conversation resolution

multi-intent processing

local context expansion

evidence grouping

evidence sufficiency and coverage checks

scope protection

answer grounding

answer guard

database/persistence components

evaluation tooling

retrieval diagnostics

extensive regression tests

Docker configuration

Generated runtime artifacts and local-machine clutter are intentionally
excluded from version control.

Important Note About Knowledge Freshness

The chatbot answers from the knowledge corpus available to it.

Institutional information can change over time, especially:

admissions

fees

schedules

regulations

contacts

events

announcements

Therefore, the chatbot should not be treated as an independent authority
when the underlying institutional source has changed.

For deployment, the knowledge-ingestion and refresh process should be
run whenever the institution requires an updated knowledge base.

Development Philosophy

The project follows a simple rule:

Correctness before cleverness.

The system is intentionally built as a chain of understandable
components rather than one opaque "magic RAG" pipeline.

The long-term objective is a chatbot that is:

accurate

grounded

conversational

scope-aware

testable

maintainable

reviewable

resilient to retrieval failures

honest about missing information

License

Add the project's official license here if/when one is formally
selected.

Maintainer

IIT Jodhpur AI Assistant / RAG Chatbot

Repository:

RAG_Chatbot

Current application version:

V3.3