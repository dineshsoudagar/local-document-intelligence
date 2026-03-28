from pathlib import Path

from src.indexing.index_service import IndexService

service = IndexService()
try:
    bundle = service.build_macro_packets("data/pdfs/RAG_survey_paper.pdf")
    print(bundle.document.title)
    print(bundle.document.page_count)
    print(len(bundle.sections))
    print("=*" * 80)

    for packet in bundle.sections:
        print(packet.section_id)
        print(f"document_heading: {bundle.document.title}")
        print(f"section_heading: {packet.section_heading}")
        print(f"pages: {packet.page_start}-{packet.page_end}")
        print(packet.section_text[:500])
        print("=" * 80)
finally:
    service.close()

