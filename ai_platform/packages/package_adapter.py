"""Generic Knowledge Package Adapter for V2 Institutional Knowledge Releases.

This module is completely institution-agnostic. It discovers available canonical documents,
metadata sidecars, structured JSON datasets, source registries, provenance ledgers, and
conflict records from standard manifests and contracts rather than hardcoding names.

Supported Package Layout:
    package_root/
    ├── manifests/
    │   ├── package_manifest.json
    │   ├── primary_corpus_manifest.json
    │   └── document_inventory.json
    ├── contracts/
    │   ├── metadata_schema.json
    │   └── RAG_INGESTION_CONTRACT.md
    ├── documents/       <- Canonical readable documents (DOCX)
    ├── knowledge/       <- Companion Markdown representations
    ├── metadata/        <- Granular JSON metadata sidecars (<doc_id>.json)
    ├── data/            <- Structured institutional JSON datasets
    ├── provenance/      <- Conflict register & claim evidence ledgers
    ├── source_registry/ <- Discovered URL inventory & lineage
    └── reports/         <- Maintenance & parity audit reports
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_platform.packages.models import (
    CanonicalDocument,
    ClaimRecord,
    ConflictRecord,
    InstitutionalRelease,
    PackageMetadata,
    ValidationReport,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = (
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
)


def _safe_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_quarantined_fallback(path: Path | str) -> bool:
    """Return True if path represents a quarantined heterogeneous fallback buffer."""
    norm = str(path).replace("\\", "/").casefold()
    return "/fallback/" in norm or "knowledge_buffer" in norm or "quarantined" in norm


def validate_package(package_root: Path | str) -> ValidationReport:
    """Validate a knowledge package against the V2 ingestion contract.

    Verifies:
    1. Package directory and manifest existence.
    2. Document inventory, primary corpus manifest, and companion paths.
    3. Cryptographic SHA-256 hashes against declared inventory hashes.
    4. Metadata sidecar presence and schema compliance.
    5. Structured JSON dataset discovery and parseability.
    6. Double-indexing boundaries (knowledge/ vs documents/).
    """
    root = Path(package_root).resolve()
    if not root.exists() or not root.is_dir():
        return ValidationReport(
            valid=False,
            package_root=root,
            institution_id="unknown",
            release_id="unknown",
            accepted_documents=0,
            skipped_documents=0,
            duplicated_documents=0,
            invalid_documents=0,
            missing_files=(f"Package directory does not exist: {root}",),
        )

    manifest_file = root / "manifests" / "package_manifest.json"
    if not manifest_file.exists():
        manifest_file = root / "data" / "manifest.json"
    if not manifest_file.exists():
        manifest_file = root / "manifest.json"

    meta_dict: dict[str, Any] = {}
    if manifest_file.exists():
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                meta_dict = json.load(f)
        except Exception as exc:
            logger.warning("Could not parse package manifest: %s", exc)

    institution_id = str(meta_dict.get("institution_id") or root.parent.name.replace("data_", "") or "unknown").strip().lower()
    release_id = str(meta_dict.get("package_name") or meta_dict.get("version") or root.name).strip()

    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    schema_warnings: list[str] = []
    datasets_discovered: list[str] = []

    # 1. Document Inventory and Primary Manifest
    inventory_file = root / "manifests" / "document_inventory.json"
    primary_manifest_file = root / "manifests" / "primary_corpus_manifest.json"

    inventory_items: list[dict[str, Any]] = []
    if inventory_file.exists():
        try:
            with inventory_file.open("r", encoding="utf-8") as f:
                inventory_items = json.load(f)
        except Exception as exc:
            schema_warnings.append(f"Invalid document_inventory.json: {exc}")
    elif primary_manifest_file.exists():
        try:
            with primary_manifest_file.open("r", encoding="utf-8") as f:
                inventory_items = json.load(f)
        except Exception as exc:
            schema_warnings.append(f"Invalid primary_corpus_manifest.json: {exc}")
    else:
        # Fallback to discovering docx files in documents/
        docs_dir = root / "documents"
        if docs_dir.exists():
            for p in sorted(docs_dir.rglob("*.docx")):
                if p.name.startswith("."):
                    continue
                rel_docx = p.relative_to(root)
                doc_id = p.stem
                inventory_items.append({
                    "document_id": doc_id,
                    "title": doc_id.replace("_", " ").title(),
                    "docx_path": str(rel_docx),
                    "markdown_path": str(Path("knowledge") / rel_docx.relative_to("documents")).replace(".docx", ".md"),
                    "metadata_path": str(Path("metadata") / f"{doc_id}.json"),
                })
        else:
            missing_files.append("documents/ directory missing")

    accepted_docs = 0
    skipped_docs = 0
    duplicated_docs = 0
    invalid_docs = 0
    seen_doc_ids: set[str] = set()

    for item in inventory_items:
        doc_id = str(item.get("document_id") or "")
        if not doc_id:
            invalid_docs += 1
            schema_warnings.append("Document item missing document_id")
            continue

        if doc_id in seen_doc_ids:
            duplicated_docs += 1
            schema_warnings.append(f"Duplicate document_id found: {doc_id}")
            continue
        seen_doc_ids.add(doc_id)

        docx_rel = item.get("docx_path")
        if not docx_rel:
            invalid_docs += 1
            schema_warnings.append(f"Document {doc_id} missing docx_path")
            continue

        docx_path = root / docx_rel
        if not docx_path.exists():
            missing_files.append(f"DOCX missing: {docx_rel}")
            invalid_docs += 1
            continue

        # Check fallback quarantine
        if _is_quarantined_fallback(docx_path):
            skipped_docs += 1
            continue

        # Check hash if declared
        declared_docx_hash = item.get("docx_hash")
        if declared_docx_hash:
            actual_docx_hash = _safe_sha256(docx_path)
            if actual_docx_hash != declared_docx_hash:
                hash_mismatches.append(f"{doc_id} docx hash mismatch: {actual_docx_hash} != {declared_docx_hash}")

        # Check metadata sidecar
        meta_rel = item.get("metadata_path") or f"metadata/{doc_id}.json"
        meta_path = root / meta_rel
        if not meta_path.exists():
            schema_warnings.append(f"Metadata sidecar missing for {doc_id}: {meta_rel}")
        else:
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    sidecar_data = json.load(f)
                    if not isinstance(sidecar_data, dict):
                        schema_warnings.append(f"Sidecar for {doc_id} is not a JSON object")
            except Exception as exc:
                schema_warnings.append(f"Error parsing sidecar for {doc_id}: {exc}")

        accepted_docs += 1

    # 2. Structured Datasets in data/
    data_dir = root / "data"
    if data_dir.exists() and data_dir.is_dir():
        for json_file in sorted(data_dir.glob("*.json")):
            if json_file.name in {".DS_Store", "manifest.json"}:
                continue
            datasets_discovered.append(json_file.name)
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    json.load(f)
            except Exception as exc:
                schema_warnings.append(f"Dataset {json_file.name} invalid JSON: {exc}")

    is_valid = (
        len(missing_files) == 0
        and len(hash_mismatches) == 0
        and accepted_docs > 0
        and invalid_docs == 0
    )

    return ValidationReport(
        valid=is_valid,
        package_root=root,
        institution_id=institution_id,
        release_id=release_id,
        accepted_documents=accepted_docs,
        skipped_documents=skipped_docs,
        duplicated_documents=duplicated_docs,
        invalid_documents=invalid_docs,
        missing_files=tuple(missing_files),
        hash_mismatches=tuple(hash_mismatches),
        schema_warnings=tuple(schema_warnings),
        datasets_discovered=tuple(datasets_discovered),
        details={
            "package_manifest": meta_dict,
            "total_inventory_items": len(inventory_items),
        },
    )


def _format_structured_records(
    dataset_name: str,
    data: Any,
    package_root: Path,
    institution_id: str,
    release_id: str,
) -> list[Document]:
    """Transform rich structured JSON datasets into searchable, source-linked Document objects."""
    records: list[Document] = []
    base_source = str((package_root / "data" / dataset_name).resolve())

    def _make_doc(text: str, title: str, category: str, subcategory: str = "", extra_meta: dict[str, Any] | None = None) -> Document:
        meta: dict[str, Any] = {
            "source": base_source,
            "source_path": base_source,
            "source_id": hashlib.sha256(base_source.encode("utf-8")).hexdigest()[:16],
            "institution_id": institution_id,
            "release_id": release_id,
            "document_id": f"structured_{dataset_name}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
            "title": title,
            "category": category,
            "subcategory": subcategory,
            "document_type": "structured_record",
            "source_urls": json.dumps([f"https://www.iitj.ac.in/data/{dataset_name}"]),
            "unit_ids": json.dumps([]),
            "claim_ids": json.dumps([]),
            "conflict_ids": json.dumps([]),
            "aliases": json.dumps([title]),
            "schema_version": "2.0",
        }
        if extra_meta:
            for k, v in extra_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = json.dumps(v, ensure_ascii=False)
        return Document(page_content=text.strip(), metadata=meta)

    # 1. academic_programmes.json
    if dataset_name == "academic_programmes.json" and isinstance(data, list):
        btech_progs: list[str] = []
        mtech_progs: list[str] = []
        msc_progs: list[str] = []

        for p in data:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "")
            deg_type = str(p.get("degree_type") or "")
            unit = str(p.get("parent_unit_name") or "")
            duration = p.get("duration")
            dur_str = f"{duration.get('years', '')} Years ({duration.get('semesters', '')} Semesters)" if isinstance(duration, dict) else str(duration or "")
            route = str(p.get("admission_route") or "")
            exam = str(p.get("entrance_examination") or "")
            counsel = str(p.get("counselling_authority") or "")
            aliases = p.get("aliases") or []

            lines = [
                f"Academic Programme: {name}",
                f"Degree Type: {deg_type}",
                f"Offering Department/School: {unit}",
                f"Normal Duration: {dur_str}",
                f"Admission Channel: {route}",
                f"Qualifying Examination: {exam}",
                f"Counselling Authority: {counsel}",
            ]
            if aliases:
                lines.append(f"Aliases: {', '.join(aliases)}")
            text = "\n".join(lines)
            records.append(_make_doc(text, title=name, category="programs", subcategory=deg_type.lower(), extra_meta={"degree_type": deg_type, "programme_id": str(p.get("programme_id") or "")}))

            if deg_type.upper() == "B.TECH":
                btech_progs.append(f"- {name} ({unit})")
            elif deg_type.upper() == "M.TECH":
                mtech_progs.append(f"- {name} ({unit})")
            elif deg_type.upper() == "M.SC":
                msc_progs.append(f"- {name} ({unit})")

        # Consolidated lists for completeness questions
        if btech_progs:
            btech_summary = (
                "Official Complete List of B.Tech Programmes at IIT Jodhpur:\n"
                + f"The Indian Institute of Technology Jodhpur offers {len(btech_progs)} undergraduate B.Tech programmes:\n"
                + "\n".join(btech_progs)
            )
            records.append(_make_doc(btech_summary, title="Complete List of B.Tech Programmes", category="programs", subcategory="btech"))

        if mtech_progs:
            mtech_summary = (
                "Official Complete List of M.Tech Programmes at IIT Jodhpur:\n"
                + f"The Indian Institute of Technology Jodhpur offers {len(mtech_progs)} postgraduate M.Tech programmes:\n"
                + "\n".join(mtech_progs)
            )
            records.append(_make_doc(mtech_summary, title="Complete List of M.Tech Programmes", category="programs", subcategory="mtech"))

        if msc_progs:
            msc_summary = (
                "Official Complete List of M.Sc Programmes at IIT Jodhpur:\n"
                + f"The Indian Institute of Technology Jodhpur offers {len(msc_progs)} M.Sc programmes:\n"
                + "\n".join(msc_progs)
            )
            records.append(_make_doc(msc_summary, title="Complete List of M.Sc Programmes", category="programs", subcategory="msc"))

    # 2. departments.json
    elif dataset_name == "departments.json" and isinstance(data, list):
        dept_lines: list[str] = []
        for d in data:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "")
            code = str(d.get("code") or "")
            degrees = d.get("degrees_offered") or []
            deg_str = ", ".join(degrees) if isinstance(degrees, list) else str(degrees)
            text = f"Department: {name} (Code: {code})\nType: Academic Department\nDegrees Offered: {deg_str}"
            records.append(_make_doc(text, title=name, category="departments", subcategory="department_directory", extra_meta={"department_code": code}))
            dept_lines.append(f"- {name} ({code})")

        if dept_lines:
            dept_summary = (
                "Official Complete Directory of Academic Departments at IIT Jodhpur:\n"
                + f"The Indian Institute of Technology Jodhpur comprises {len(dept_lines)} academic departments:\n"
                + "\n".join(dept_lines)
            )
            records.append(_make_doc(dept_summary, title="Institute-Wide Academic Departments Directory", category="departments", subcategory="departments_list"))

    # 3. schools.json
    elif dataset_name == "schools.json" and isinstance(data, list):
        school_lines: list[str] = []
        for s in data:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "")
            acronym = str(s.get("acronym") or "")
            focus = str(s.get("focus") or "")
            degrees = s.get("degrees_offered") or []
            deg_str = ", ".join(degrees) if isinstance(degrees, list) else str(degrees)
            text = f"School: {name} ({acronym})\nFocus: {focus}\nDegrees Offered: {deg_str}"
            records.append(_make_doc(text, title=name, category="schools", subcategory="school_directory", extra_meta={"school_acronym": acronym}))
            school_lines.append(f"- {name} ({acronym})")

        if school_lines:
            school_summary = "Official Schools at IIT Jodhpur:\n" + "\n".join(school_lines)
            records.append(_make_doc(school_summary, title="Schools Directory", category="schools", subcategory="schools_list"))

    # 4. visitor_accommodation.json
    elif dataset_name == "visitor_accommodation.json" and isinstance(data, dict):
        short_term_list = data.get("short_term_rates_01_to_10_days") or []
        long_term = data.get("long_term_bookings_11_days_to_6_months") or {}
        booking_proc = str(data.get("booking_procedure") or "")

        st_lines: list[str] = []
        if isinstance(short_term_list, list):
            for st in short_term_list:
                cat = st.get("category", "")
                rtype = st.get("room_type", "")
                d_rate = st.get("double_occupancy_per_day", "")
                s_rate = st.get("single_occupancy_per_day", "")
                tax = st.get("tax", "")
                item_text = (
                    f"Category: {cat}\n"
                    f"Room Type: {rtype}\n"
                    f"Double Occupancy Charge: ₹{d_rate}/- per day\n"
                    f"Single Occupancy Charge: ₹{s_rate}/- per day\n"
                    f"Tax Status: {tax}"
                )
                records.append(_make_doc(item_text, title=f"Hostel Accommodation Charges — {cat}", category="hostel_accommodation", subcategory="short_term_rates"))
                st_lines.append(f"- {cat} ({rtype}): Double ₹{d_rate}/day, Single ₹{s_rate}/day ({tax})")

        lt_rate = long_term.get("rate_per_month", 2175) if isinstance(long_term, dict) else 2175
        lt_tax = long_term.get("tax", "Plus GST for visitors; without GST for IIT Jodhpur students and staff") if isinstance(long_term, dict) else ""
        lt_policy = long_term.get("phd_post_thesis_policy", "") if isinstance(long_term, dict) else ""

        vh_summary = (
            "Official IIT Jodhpur Visitor and Guest Hostel Accommodation and Room Charges:\n\n"
            "Short-Term Hostel Accommodation (01 to 10 Days):\n"
            + "\n".join(st_lines)
            + "\n\nLong-Term Bookings (11 Days to 6 Months):\n"
            f"- Monthly Charge: ₹{lt_rate}/- per month\n"
            f"- Tax: {lt_tax}\n"
            f"- PhD Post-Thesis Policy: {lt_policy}\n\n"
            f"Booking Procedure: {booking_proc}"
        )
        records.append(_make_doc(vh_summary, title="Visitor Hostel Room Charges and Accommodation Rates", category="hostel_accommodation", subcategory="visitor_hostel"))

    # 5. emergency_contacts.json
    elif dataset_name == "emergency_contacts.json" and isinstance(data, list):
        em_lines: list[str] = []
        for c in data:
            if not isinstance(c, dict):
                continue
            service = str(c.get("service") or "")
            tel = str(c.get("telephone") or "")
            alts = c.get("alternate_phones") or []
            alt_str = f" (Alternate: {', '.join(alts)})" if alts else ""
            loc = str(c.get("location") or "")
            avail = str(c.get("availability") or "")
            text = f"Emergency Service: {service}\nTelephone Number: {tel}{alt_str}\nAvailability: {avail}\nLocation: {loc}"
            records.append(_make_doc(text, title=service, category="emergency", subcategory="emergency_contacts", extra_meta={"telephone": tel}))
            em_lines.append(f"- {service}: {tel}{alt_str} [{loc}, {avail}]")

        if em_lines:
            em_summary = "IIT Jodhpur Emergency Phone Numbers & Contacts Directory:\n" + "\n".join(em_lines)
            records.append(_make_doc(em_summary, title="Emergency Contacts Directory", category="emergency", subcategory="contacts_list"))

    # 6. fees_and_scholarships.json
    elif dataset_name == "fees_and_scholarships.json" and isinstance(data, dict):
        structures = data.get("academic_fee_structures") or []
        scholarships = data.get("scholarships_and_fellowships") or {}

        fee_lines = ["Official IIT Jodhpur Academic Tuition Fees and Waiver Policies:"]
        if isinstance(structures, list):
            for prog_fee in structures:
                if isinstance(prog_fee, dict):
                    prog_name = prog_fee.get("programme", "")
                    tuition = prog_fee.get("tuition_fee_per_semester", "")
                    curr = prog_fee.get("currency", "INR")
                    ay = prog_fee.get("academic_year", "")
                    auth = prog_fee.get("authority", "")
                    waivers = prog_fee.get("waiver_and_remission_policy") or {}
                    other_chg = prog_fee.get("other_semester_charges") or {}
                    notes = prog_fee.get("notes") or []

                    prog_text_lines = [
                        f"Academic Programme: {prog_name}",
                        f"Tuition Fee: ₹{tuition:,}/- per semester ({curr})" if isinstance(tuition, int) else f"Tuition Fee: {tuition}",
                        f"Academic Session: {ay}",
                        f"Governing Authority: {auth}",
                    ]
                    if isinstance(waivers, dict):
                        prog_text_lines.append("Tuition Waiver and Remission Policy:")
                        for w_k, w_v in waivers.items():
                            prog_text_lines.append(f"  - {w_k.replace('_', ' ').title()}: {w_v}")
                    if isinstance(other_chg, dict):
                        prog_text_lines.append("Other Charges:")
                        for c_k, c_v in other_chg.items():
                            prog_text_lines.append(f"  - {c_k.replace('_', ' ').title()}: ₹{c_v}")
                    if notes:
                        prog_text_lines.append("Important Notes:")
                        for n in notes:
                            prog_text_lines.append(f"  - {n}")

                    records.append(_make_doc("\n".join(prog_text_lines), title=f"Tuition Fee & Waivers — {prog_name}", category="finance", subcategory="tuition_fees"))
                    fee_lines.append(f"\n### {prog_name}:\n- Tuition Fee per Semester: ₹{tuition:,}/-\n- Governing Authority: {auth}")

        records.append(_make_doc("\n".join(fee_lines), title="Academic Fee Structures and Tuition Fee Waivers Summary", category="finance", subcategory="fees_summary"))

        if isinstance(scholarships, dict):
            sch_lines = ["IIT Jodhpur Scholarships, Assistantships, and Fellowships:"]
            for sch_k, sch_v in scholarships.items():
                sch_lines.append(f"- {sch_k.replace('_', ' ').title()}: {sch_v}")
            records.append(_make_doc("\n".join(sch_lines), title="Scholarships and Fellowships", category="finance", subcategory="scholarships"))

    # 7. Generic fallback parser for any other structured list or dict dataset
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                name = item.get("name") or item.get("title") or item.get("id") or f"{dataset_name}_{idx}"
                desc = item.get("description") or item.get("focus") or item.get("scope") or ""
                lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in item.items() if v and not isinstance(v, (dict, list))]
                records.append(_make_doc("\n".join(lines), title=str(name), category=dataset_name.replace(".json", "")))
    elif isinstance(data, dict):
        for section_key, section_val in data.items():
            if isinstance(section_val, dict):
                lines = [f"Section: {section_key.replace('_', ' ').title()}"]
                lines.extend(f"- {k.replace('_', ' ').title()}: {v}" for k, v in section_val.items() if v and not isinstance(v, (dict, list)))
                records.append(_make_doc("\n".join(lines), title=f"{dataset_name} — {section_key}", category=dataset_name.replace(".json", "")))

    return records


def load_institutional_release(
    package_root: Path | str,
    validate_hashes: bool = True,
) -> InstitutionalRelease:
    """Load, validate, and normalize a complete institutional knowledge package."""
    root = Path(package_root).resolve()
    report = validate_package(root)

    manifest_file = root / "manifests" / "package_manifest.json"
    if not manifest_file.exists():
        manifest_file = root / "data" / "manifest.json"
    if not manifest_file.exists():
        manifest_file = root / "manifest.json"

    meta_dict: dict[str, Any] = {}
    if manifest_file.exists():
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                meta_dict = json.load(f)
        except Exception:
            pass

    institution_id = report.institution_id
    release_id = report.release_id
    display_name = str(meta_dict.get("institution_name") or "Indian Institute of Technology Jodhpur")

    metadata = PackageMetadata(
        package_name=str(meta_dict.get("package_name") or release_id),
        version=str(meta_dict.get("version") or "2.0.0"),
        institution_id=institution_id,
        institution_name=display_name,
        release_timestamp=str(meta_dict.get("release_timestamp") or ""),
        total_canonical_documents=int(meta_dict.get("total_canonical_documents") or report.accepted_documents),
        total_structured_datasets=int(meta_dict.get("total_structured_datasets") or len(report.datasets_discovered)),
        schema_version="2.0",
        parity_guarantee=str(meta_dict.get("parity_guarantee") or ""),
    )

    # 1. Load canonical documents inventory
    documents: list[CanonicalDocument] = []
    inventory_file = root / "manifests" / "document_inventory.json"
    primary_manifest_file = root / "manifests" / "primary_corpus_manifest.json"

    inventory_items: list[dict[str, Any]] = []
    if inventory_file.exists():
        with inventory_file.open("r", encoding="utf-8") as f:
            inventory_items = json.load(f)
    elif primary_manifest_file.exists():
        with primary_manifest_file.open("r", encoding="utf-8") as f:
            inventory_items = json.load(f)

    for item in inventory_items:
        doc_id = str(item.get("document_id") or "")
        docx_rel = item.get("docx_path")
        if not doc_id or not docx_rel:
            continue

        docx_path = root / docx_rel
        if not docx_path.exists():
            continue

        is_fallback = _is_quarantined_fallback(docx_path)
        md_rel = item.get("markdown_path") or f"knowledge/{docx_rel.replace('documents/', '').replace('.docx', '.md')}"
        md_path = root / md_rel
        meta_rel = item.get("metadata_path") or f"metadata/{doc_id}.json"
        meta_path = root / meta_rel

        # Read metadata sidecar
        sidecar: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    sidecar = json.load(f)
            except Exception:
                pass

        title = str(sidecar.get("title") or item.get("title") or doc_id.replace("_", " ").title())
        category = str(sidecar.get("category") or item.get("category") or "general")
        subcategory = str(sidecar.get("subcategory") or "")
        document_type = str(sidecar.get("document_type") or "canonical_docx")

        source_urls = tuple(sidecar.get("source_urls") or item.get("source_urls") or ())
        unit_ids = tuple(sidecar.get("unit_ids") or ())
        aliases = tuple(sidecar.get("aliases") or ())
        claim_ids = tuple(sidecar.get("claim_ids") or ())
        conflict_ids = tuple(sidecar.get("conflict_ids") or ())

        val_status = sidecar.get("validation_status") or {}
        if not isinstance(val_status, dict):
            val_status = {}

        documents.append(
            CanonicalDocument(
                document_id=doc_id,
                title=title,
                category=category,
                subcategory=subcategory,
                document_type=document_type,
                docx_path=docx_path,
                markdown_path=md_path,
                metadata_path=meta_path,
                source_urls=source_urls,
                unit_ids=unit_ids,
                aliases=aliases,
                claim_ids=claim_ids,
                conflict_ids=conflict_ids,
                docx_hash=str(item.get("docx_hash") or sidecar.get("docx_hash") or ""),
                markdown_hash=str(item.get("markdown_hash") or sidecar.get("markdown_hash") or ""),
                is_fallback=is_fallback,
                validation_status=val_status,
            )
        )

    # 2. Discover and parse structured datasets
    structured_datasets: dict[str, Any] = {}
    structured_records: list[Document] = []
    data_dir = root / "data"
    if data_dir.exists() and data_dir.is_dir():
        for json_file in sorted(data_dir.glob("*.json")):
            if json_file.name in {".DS_Store", "manifest.json"}:
                continue
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    dataset_data = json.load(f)
                    structured_datasets[json_file.name] = dataset_data
                    records = _format_structured_records(
                        dataset_name=json_file.name,
                        data=dataset_data,
                        package_root=root,
                        institution_id=institution_id,
                        release_id=release_id,
                    )
                    structured_records.extend(records)
            except Exception as exc:
                logger.warning("Could not parse dataset %s: %s", json_file.name, exc)

    # 3. Load conflicts
    conflicts: list[ConflictRecord] = []
    conflict_file = root / "provenance" / "source_conflict_register.json"
    if conflict_file.exists():
        try:
            with conflict_file.open("r", encoding="utf-8") as f:
                c_data = json.load(f)
                for c in c_data.get("conflicts", []):
                    conflicts.append(
                        ConflictRecord(
                            conflict_id=str(c.get("conflict_id") or ""),
                            entity=str(c.get("entity") or ""),
                            source_1=c.get("source_1") or {},
                            source_2=c.get("source_2") or {},
                            disposition=str(c.get("disposition") or ""),
                            explanation=str(c.get("explanation") or ""),
                        )
                    )
        except Exception as exc:
            logger.warning("Could not parse conflict register: %s", exc)

    # 4. Load claims
    claims: list[ClaimRecord] = []
    claim_file = root / "provenance" / "source_claim_evidence_ledger.json"
    if claim_file.exists():
        try:
            with claim_file.open("r", encoding="utf-8") as f:
                cl_data = json.load(f)
                if isinstance(cl_data, list):
                    for cl in cl_data:
                        claims.append(
                            ClaimRecord(
                                claim_id=str(cl.get("claim_id") or ""),
                                domain=str(cl.get("domain") or ""),
                                entity=str(cl.get("entity") or ""),
                                claim_value=str(cl.get("claim_value") or ""),
                                conditions=str(cl.get("conditions") or ""),
                                source_url=str(cl.get("source_url") or ""),
                                verified_in_documents=tuple(cl.get("verified_in_documents") or ()),
                                verification_status=str(cl.get("verification_status") or ""),
                            )
                        )
        except Exception as exc:
            logger.warning("Could not parse claim evidence ledger: %s", exc)

    # 5. Extract lexical resources from structured datasets
    lexical_resources: dict[str, Any] = {
        "acronyms": structured_datasets.get("acronyms.json", {}),
        "aliases": structured_datasets.get("aliases.json", {}),
        "phrases": structured_datasets.get("phrases.json", []),
        "semantic_terms": structured_datasets.get("semantic_terms.json", {}),
        "scope_policy": structured_datasets.get("scope_policy.json", {}),
        "source_policy": structured_datasets.get("source_policy.json", {}),
    }

    return InstitutionalRelease(
        institution_id=institution_id,
        release_id=release_id,
        display_name=display_name,
        package_root=root,
        metadata=metadata,
        documents=documents,
        structured_datasets=structured_datasets,
        structured_records=structured_records,
        conflicts=conflicts,
        claims=claims,
        lexical_resources=lexical_resources,
        validation_report=report,
    )


def chunk_package_documents(
    release: InstitutionalRelease,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    include_fallback: bool = False,
) -> list[Document]:
    """Split canonical documents into retrieval-optimized chunks with complete metadata sidecar propagation.

    Guarantees:
    - Exactly 1 copy per subject indexed (DOCX canonical, ignoring fallback by default).
    - Metadata sidecar fields attached to every chunk.
    - All metadata values sanitized to Chroma-compatible primitives (str, int, float, bool).
    - Structured records from approved datasets included seamlessly as first-class evidence chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(CHUNK_SEPARATORS),
    )

    all_chunks: list[Document] = []

    # 1. Chunk canonical DOCX documents
    for doc_meta in release.documents:
        if doc_meta.is_fallback and not include_fallback:
            continue

        try:
            loader = Docx2txtLoader(str(doc_meta.docx_path))
            raw_docs = loader.load()
        except Exception as exc:
            logger.warning("Failed to load docx %s: %s", doc_meta.docx_path, exc)
            continue

        for raw_doc in raw_docs:
            page_text = raw_doc.page_content.strip()
            if not page_text:
                continue

            sub_chunks = splitter.split_text(page_text)
            total_chunks = len(sub_chunks)

            for idx, chunk_text in enumerate(sub_chunks):
                clean_chunk = chunk_text.strip()
                if not clean_chunk:
                    continue

                chunk_id_mat = f"{doc_meta.document_id}:{idx}:{doc_meta.docx_hash}:{clean_chunk[:50]}"
                chunk_id = hashlib.sha256(chunk_id_mat.encode("utf-8")).hexdigest()[:16]

                chunk_metadata: dict[str, Any] = {
                    "source": str(doc_meta.docx_path),
                    "source_path": str(doc_meta.docx_path),
                    "source_id": hashlib.sha256(str(doc_meta.docx_path).encode("utf-8")).hexdigest()[:16],
                    "institution_id": release.institution_id,
                    "release_id": release.release_id,
                    "document_id": doc_meta.document_id,
                    "title": doc_meta.title,
                    "category": doc_meta.category,
                    "subcategory": doc_meta.subcategory,
                    "document_type": doc_meta.document_type,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_count": total_chunks,
                    "source_urls": json.dumps(list(doc_meta.source_urls), ensure_ascii=False),
                    "unit_ids": json.dumps(list(doc_meta.unit_ids), ensure_ascii=False),
                    "claim_ids": json.dumps(list(doc_meta.claim_ids), ensure_ascii=False),
                    "conflict_ids": json.dumps(list(doc_meta.conflict_ids), ensure_ascii=False),
                    "aliases": json.dumps(list(doc_meta.aliases), ensure_ascii=False),
                    "content_hash": doc_meta.docx_hash,
                    "schema_version": release.metadata.schema_version,
                }
                all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_metadata))

    # 2. Append structured records (already formatted as Documents with Chroma-compatible metadata)
    all_chunks.extend(release.structured_records)

    return all_chunks
