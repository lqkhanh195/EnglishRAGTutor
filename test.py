# import os
# from llama_index.core import SimpleDirectoryReader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.node_parser import SemanticSplitterNodeParser

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# reader = SimpleDirectoryReader(input_dir="knowledge_base\\ANHTC 1-2\\Tactics for TOEIC (Speaking and Writing)")
# documents = reader.load_data(show_progress=True)

# splitter = SemanticSplitterNodeParser(
#     buffer_size=1,
#     breakpoint_percentile_threshold=98,
#     embed_model=embed_model
# )

# nodes = splitter.get_nodes_from_documents(documents)
# # for node in nodes:
# #     print(node.text)

# with open("test.txt", "w", encoding="utf-8") as f:
#     for node in nodes:
#         f.write(node.text)
#         f.write("\n==============================\n")

# import logging
# from tika import parser
# import re
# from io import BytesIO

# from deepdoc.parser.utils import get_text
# from rag.nlp import bullets_category, is_english,remove_contents_table, \
#     hierarchical_merge, make_colon_as_title, naive_merge, random_choices, tokenize_table, \
#     tokenize_chunks
# from rag.nlp import rag_tokenizer
# from deepdoc.parser import PdfParser, PlainParser


# class Pdf(PdfParser):
#     def __call__(self, filename, binary=None, from_page=0,
#                  to_page=100000, zoomin=3, callback=None):
#         from timeit import default_timer as timer
#         start = timer()
#         callback(msg="OCR started")
#         self.__images__(
#             filename if not binary else binary,
#             zoomin,
#             from_page,
#             to_page,
#             callback)
#         callback(msg="OCR finished ({:.2f}s)".format(timer() - start))

#         start = timer()
#         self._layouts_rec(zoomin)
#         callback(0.67, "Layout analysis ({:.2f}s)".format(timer() - start))
#         logging.debug("layouts: {}".format(timer() - start))

#         start = timer()
#         self._table_transformer_job(zoomin)
#         callback(0.68, "Table analysis ({:.2f}s)".format(timer() - start))

#         start = timer()
#         self._text_merge()
#         tbls = self._extract_table_figure(True, zoomin, True, True)
#         self._naive_vertical_merge()
#         self._filter_forpages()
#         self._merge_with_same_bullet()
#         callback(0.8, "Text extraction ({:.2f}s)".format(timer() - start))

#         return [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", ""))
#                 for b in self.boxes], tbls


# def chunk(filename, binary=None, from_page=0, to_page=100000,
#           lang="Chinese", callback=None, **kwargs):
#     """
#         Supported file formats are docx, pdf, txt.
#         Since a book is long and not all the parts are useful, if it's a PDF,
#         please setup the page ranges for every book in order eliminate negative effects and save elapsed computing time.
#     """
#     doc = {
#         "docnm_kwd": filename,
#         "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
#     }
#     doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
#     pdf_parser = None
#     sections, tbls = [], []

#     pdf_parser = Pdf() if kwargs.get(
#         "parser_config", {}).get(
#         "layout_recognize", True) else PlainParser()
#     sections, tbls = pdf_parser(filename if not binary else binary,
#                                 from_page=from_page, to_page=to_page, callback=callback)

#     make_colon_as_title(sections)
#     bull = bullets_category(
#         [t for t in random_choices([t for t, _ in sections], k=100)])
#     if bull >= 0:
#         chunks = ["\n".join(ck)
#                   for ck in hierarchical_merge(bull, sections, 5)]
#     else:
#         sections = [s.split("@") for s, _ in sections]
#         sections = [(pr[0], "@" + pr[1]) if len(pr) == 2 else (pr[0], '') for pr in sections ]
#         chunks = naive_merge(
#             sections, kwargs.get(
#                 "chunk_token_num", 256), kwargs.get(
#                 "delimer", "\n。；！？"))

#     # is it English
#     # is_english(random_choices([t for t, _ in sections], k=218))
#     eng = lang.lower() == "english"

#     res = tokenize_table(tbls, doc, eng)
#     res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))

#     return res


# if __name__ == "__main__":
#     import sys

#     def dummy(prog=None, msg=""):
#         pass
#     # chunk(sys.argv[1], from_page=1, to_page=10, callback=dummy)

#     print(chunk("D:\\EngRAG\\LiteratureIsEasy\\knowledge_base\\ANHTC 1-2\\Tactics for TOEIC (Speaking and Writing)\\Sample_tests.pdf", from_page=0, to_page=100000, lang="English", callback=dummy))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:\\EngRAG\\LiteratureIsEasy\\knowledge_base\\ANHTC 1-2\\Tactics for TOEIC (Speaking and Writing)\\Sample_tests.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(docs)

with open("test.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk.page_content)
        f.write("==========================")

