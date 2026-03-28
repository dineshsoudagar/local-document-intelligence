from pathlib import Path

from src.indexing.index_service import IndexService

service = IndexService()
try:
    bundle = service.build_macro_packets("data/pdfs/RAG_survey_paper.pdf")
    print(bundle.document.title)
    print(bundle.document.page_count)
    print(len(bundle.sections))

    for section in bundle.sections[:5]:
        print("=" * 80)
        print(section.section_id)
        print(section.display_heading)
        print(section.page_start, section.page_end)
        print(section.chunk_count)
        for excerpt in section.representative_excerpts:
            print("-", excerpt.rationale, excerpt.page_start, excerpt.page_end)
            print(excerpt.text[:200])
finally:
    service.close()

