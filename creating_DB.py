import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Open the PDF
pdf_document = fitz.open('/Users/brototodeb/Desktop/llm-examples_2/MGH White Book Housestaff Manual 2023-2024.pdf')
# Check the number of pages
print("Number of pages: ", pdf_document.page_count)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0,length_function = len)

all_gl_pages = []
for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text("text")
        all_gl_pages.append({'text':page_text, 'page_num':page_num})


# Convert all_gl_pages list to a list of objects with a page_content attribute
class PageObject:
    def __init__(self, page_content,metadata):
        self.page_content = page_content
        self.metadata = metadata

all_gl_pages_objects = [PageObject(i['text'],{'content':'MGH handbook','page_num':i['page_num']}) for i in all_gl_pages]

# Split the documents using the text_splitter
splits = text_splitter.split_documents(all_gl_pages_objects)

from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://5fff5f2d-b0f0-4ecc-aefa-81905ce94dc9.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="YYDLXtDyw75MKxpErAvcIqkGPxo_66qZILNb-EDLeoFJPgi8LbdKFQ",
)

qdrant_vectorstore = Qdrant.from_documents(splits,
    base_embeddings,url="https://5fff5f2d-b0f0-4ecc-aefa-81905ce94dc9.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="YYDLXtDyw75MKxpErAvcIqkGPxo_66qZILNb-EDLeoFJPgi8LbdKFQ",
    collection_name="rag_tech_db",
    force_recreate=False
)