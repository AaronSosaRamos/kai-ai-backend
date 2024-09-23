from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader, TextLoader, UnstructuredURLLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredXMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from app.utils.allowed_file_extensions import FileType
from app.api.error_utilities import FileHandlerError, ImageHandlerError,ImageSummaryHandlerError
from app.api.error_utilities import VideoTranscriptError
from langchain_core.messages import HumanMessage
from app.services.logger import setup_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import tempfile
import uuid
import requests
import gdown
import shutil
import google.generativeai as genai
from typing import Any
import os
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import cv2
from pytubefix import YouTube
from pytubefix.cli import on_progress

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

from unstructured.partition.pdf import partition_pdf

STRUCTURED_TABULAR_FILE_EXTENSIONS = {"csv", "xls", "xlsx", "gsheet", "xml"}

logger = setup_logger(__name__)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)

    with open(absolute_file_path, 'r') as file:
        return file.read()

def get_docs(file_url: str, file_type: str, verbose=True):
    print(file_type)
    file_type = file_type.lower()
    try:
        file_loader = file_loader_map[FileType(file_type)]
        docs = file_loader(file_url, verbose)
        print(docs)
        return docs

    except Exception as e:
        print(e)
        logger.error(f"Unsupported file type: {file_type}")
        raise FileHandlerError(f"Unsupported file type", file_url) from e
    
#Helper method to create summaries of the images extracted from pdfs and videos
def generate_image_summaries(img_dir):
    img_docs = []

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    prompt = read_text_file(os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/prompt/img_prompt.txt"))

    for filename in sorted(os.listdir(img_dir)):
      print(filename)
      try:
       if filename.endswith(".jpg"):
        image_path = os.path.join(img_dir,filename)
        img = Image.open(image_path)
        response = model.generate_content([prompt,img])
        image_summary = response.candidates[0].content.parts[0].text

        img_doc = Document(page_content=image_summary,metadata={"image_url":image_path})
        img_docs.append(img_doc)

      except Exception as e:
        logger.error(f"No such file found at {image_path}")
        raise ImageSummaryHandlerError(f"The file is not an image and cannot be summarised.", {image_path}) from e
    logger.info("Image summaries generated!")
    return img_docs

class FileHandler:
    def __init__(self, file_loader, file_extension):
        self.file_loader = file_loader
        self.file_extension = file_extension

    def load(self, url):
        # Generate a unique filename with a UUID prefix
        unique_filename = f"{uuid.uuid4()}.{self.file_extension}"

        # Download the file from the URL and save it to a temporary file
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        with tempfile.NamedTemporaryFile(delete=False, prefix=unique_filename) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Use the file_loader to load the documents
        try:
            loader = self.file_loader(file_path=temp_file_path)
        except Exception as e:
            logger.error(f"No such file found at {temp_file_path}")
            raise FileHandlerError(f"No file found", temp_file_path) from e

        try:
            dir_name = os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/images")
            os.makedirs(dir_name)

            dir = (os.path.dirname(os.path.abspath("./images")))
            documents = loader.load()
            if self.file_extension == "pdf":
              partition_pdf(
               filename=temp_file_path,
               extract_images_in_pdf=True,
               infer_table_structure=True,
               chunking_strategy="by_title",
               max_characters=4000,
               new_after_n_chars=3800,
               combine_text_under_n_chars=2000,
               extract_image_block_output_dir= os.path.join(dir, "app/features/quizzify/images"))
              
        except Exception as e:
            logger.error(f"File content might be private or unavailable or the URL is incorrect.")
            raise FileHandlerError(f"No file content available", temp_file_path) from e

        # Remove the temporary file
        os.remove(temp_file_path)

        return documents


def load_pdf_documents(pdf_url: str, verbose=False):
    pdf_loader = FileHandler(PyPDFLoader, "pdf")
    docs = pdf_loader.load(pdf_url)

    #directory that stores the images
    image_dir =os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/images")

    #extracting images from the pdf and storing them in a directory
    image_docs = generate_image_summaries(image_dir)

    if docs:
        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found PDF file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

    shutil.rmtree(image_dir)
    return image_docs + docs


def load_csv_documents(csv_url: str, verbose=False):
    csv_loader = FileHandler(CSVLoader, "csv")
    docs = csv_loader.load(csv_url)

    if docs:
        if verbose:
            logger.info(f"Found CSV file")
            logger.info(f"Splitting documents into {len(docs)} chunks")

        return docs

def load_txt_documents(notes_url: str, verbose=False):
    notes_loader = FileHandler(TextLoader, "txt")
    docs = notes_loader.load(notes_url)

    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found TXT file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_md_documents(notes_url: str, verbose=False):
    notes_loader = FileHandler(TextLoader, "md")
    docs = notes_loader.load(notes_url)

    if docs:

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found MD file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_url_documents(url: str, verbose=False):
    url_loader = UnstructuredURLLoader(urls=[url])
    docs = url_loader.load()

    if docs:
        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found URL")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_pptx_documents(pptx_url: str, verbose=False):
    pptx_handler = FileHandler(UnstructuredPowerPointLoader, 'pptx')

    docs = pptx_handler.load(pptx_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found PPTX file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_docx_documents(docx_url: str, verbose=False):
    docx_handler = FileHandler(Docx2txtLoader, 'docx')
    docs = docx_handler.load(docx_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found DOCX file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_xls_documents(xls_url: str, verbose=False):
    xls_handler = FileHandler(UnstructuredExcelLoader, 'xls')
    docs = xls_handler.load(xls_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found XLS file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_xlsx_documents(xlsx_url: str, verbose=False):
    xlsx_handler = FileHandler(UnstructuredExcelLoader, 'xlsx')
    docs = xlsx_handler.load(xlsx_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found XLSX file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_xml_documents(xml_url: str, verbose=False):
    xml_handler = FileHandler(UnstructuredXMLLoader, 'xml')
    docs = xml_handler.load(xml_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found XML file")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs


class FileHandlerForGoogleDrive:
    def __init__(self, file_loader, file_extension='docx'):
        self.file_loader = file_loader
        self.file_extension = file_extension

    def load(self, url):

        unique_filename = f"{uuid.uuid4()}.{self.file_extension}"

        try:
            gdown.download(url=url, output=unique_filename, fuzzy=True)
        except Exception as e:
            logger.error(f"File content might be private or unavailable or the URL is incorrect.")
            raise FileHandlerError(f"No file content available") from e

        try:
            loader = self.file_loader(file_path=unique_filename)
        except Exception as e:
            logger.error(f"No such file found at {unique_filename}")
            raise FileHandlerError(f"No file found", unique_filename) from e

        try:
            documents = loader.load()
        except Exception as e:
            logger.error(f"File content might be private or unavailable or the URL is incorrect.")
            raise FileHandlerError(f"No file content available") from e

        os.remove(unique_filename)

        return documents

def load_gdocs_documents(drive_folder_url: str, verbose=False):

    gdocs_loader = FileHandlerForGoogleDrive(Docx2txtLoader)

    docs = gdocs_loader.load(drive_folder_url)

    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found Google Docs files")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_gsheets_documents(drive_folder_url: str, verbose=False):
    gsheets_loader = FileHandlerForGoogleDrive(UnstructuredExcelLoader, 'xlsx')
    docs = gsheets_loader.load(drive_folder_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found Google Sheets files")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_gslides_documents(drive_folder_url: str, verbose=False):
    gslides_loader = FileHandlerForGoogleDrive(UnstructuredPowerPointLoader, 'pptx')
    docs = gslides_loader.load(drive_folder_url)
    if docs: 

        split_docs = splitter.split_documents(docs)

        if verbose:
            logger.info(f"Found Google Slides files")
            logger.info(f"Splitting documents into {len(split_docs)} chunks")

        return split_docs

def load_gpdf_documents(drive_folder_url: str, verbose=False):

    gpdf_loader = FileHandlerForGoogleDrive(PyPDFLoader,'pdf')

    docs = gpdf_loader.load(drive_folder_url)
    if docs: 

        if verbose:
            logger.info(f"Found Google PDF files")
            logger.info(f"Splitting documents into {len(docs)} chunks")

        return docs


def load_docs_youtube_url(youtube_url: str, verbose=True) -> str:
    #Extracting key image frames from the video and processing them to document objects
    yt = YouTube(youtube_url,on_progress_callback=on_progress)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path = os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/video_data/videos"))

    dir_name = os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/video_data/videos")
    extract_image_frames(dir_name)

    img_docs = generate_image_summaries(os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/video_data/video_img"))
       
    shutil.rmtree("./app/features/quizzify/video_data")
    
    #Extracting the audio files and processing them to document objects
    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
    docs = loader.load()
    audio_docs = splitter.split_documents(docs)

    return img_docs + audio_docs


def extract_image_frames(vid_dir):
    output_folder= os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/video_data/video_img")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in sorted(os.listdir(vid_dir)):
        file_path = os.path.join(vid_dir,filename)
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps*10)

        count = 0
        extracted_frames = 0

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES,count)
            ret,frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(output_folder,f"frame_{extracted_frames}.jpg")
            cv2.imwrite(frame_filename,frame)
            print(f"saved: {frame_filename}")
            count+=frame_interval
            extracted_frames+=1
    


llm_for_img = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def generate_docs_from_img(img_file_path,img_url, verbose: bool=False):
    message = HumanMessage(
    content=[
            {
                "type": "text",
                "text": "Give me a summary of what you see in the image. It must be 3 detailed paragraphs.",
            }, 
            {"type": "image_url", "image_url": img_url},
        ]
    )

    try:
        response = llm_for_img.invoke([message]).content
        logger.info(f"Generated summary: {response}")
        docs = Document(page_content=response, metadata={"source": img_url})
        split_docs = splitter.split_documents([docs])
    except Exception as e:
        logger.error(f"Error processing the request due to Invalid Content or Invalid Image URL")
        raise ImageHandlerError(f"Error processing the request", img_url) from e

    return split_docs

# from IPython.display import Image


file_loader_map = {
    FileType.PDF: load_pdf_documents,
    FileType.CSV: load_csv_documents,
    FileType.TXT: load_txt_documents,
    FileType.MD: load_md_documents,
    FileType.URL: load_url_documents,
    FileType.PPTX: load_pptx_documents,
    FileType.DOCX: load_docx_documents,
    FileType.XLS: load_xls_documents,
    FileType.XLSX: load_xlsx_documents,
    FileType.XML: load_xml_documents,
    FileType.GDOC: load_gdocs_documents,
    FileType.GSHEET: load_gsheets_documents,
    FileType.GSLIDE: load_gslides_documents,
    FileType.GPDF: load_gpdf_documents,
    FileType.YOUTUBE_URL: load_docs_youtube_url,
    FileType.IMG: generate_docs_from_img
}