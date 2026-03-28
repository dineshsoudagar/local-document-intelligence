from dataclasses import asdict, fields, is_dataclass
from pprint import pprint

from src.indexing.index_service import IndexService


def print_keys(title: str, obj) -> None:
    print(title)
    print("-" * len(title))

    if not is_dataclass(obj):
        print("Not a dataclass")
        return

    print("field names:")
    print([field.name for field in fields(obj)])
    print()

    print("values:")
    pprint(asdict(obj), sort_dicts=False, width=140)
    print()


service = IndexService()

try:
    bundle = service.build_macro_packets("data/pdfs/RAG_survey_paper.pdf")

    print_keys("BUNDLE", bundle)
    print_keys("DOCUMENT", bundle.document)

    print("DOCUMENT DERIVED")
    print("----------------")
    print(f"title: {bundle.document.title}")
    print(f"page_count: {bundle.document.page_count}")
    print(f"chunk_count: {bundle.document.chunk_count}")
    print(f"section_count: {len(bundle.sections)}")
    print("=" * 100)

    for index, packet in enumerate(bundle.sections[:5], start=1):
        print_keys(f"SECTION {index}", packet)

        print("SECTION DERIVED")
        print("---------------")
        print(f"section_id: {packet.section_id}")
        print(f"section_heading: {packet.section_heading}")
        print(f"document_heading: {bundle.document.title}")
        print(f"pages: {packet.page_start}-{packet.page_end}")
        print("section_text_preview:")
        print(packet.section_text[:800])
        print("=" * 100)

finally:
    service.close()