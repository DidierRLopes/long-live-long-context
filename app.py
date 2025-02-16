import streamlit as st
import os
import google.generativeai as genai
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import time
import tempfile

st.header("Long live long context (Gemini 2.0 Flash)")

# Configure page
st.set_page_config(page_title="Long live long context (Gemini 2.0 Flash)", layout="wide")

# API Key input
api_key = st.text_input("Enter Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

# ------------------------------- System prompt section -------------------------------
st.header("System Prompt")
system_prompt_path = "data/system_prompt.txt"
try:
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
except:
    system_prompt = ""

# Initialize editing state in session state if not present
if "editing_system_prompt" not in st.session_state:
    st.session_state.editing_system_prompt = False

# Display edit button
if not st.session_state.editing_system_prompt:
    st.text_area("System prompt", value=system_prompt, disabled=True, height=300)
    if st.button("Edit"):
        st.session_state.editing_system_prompt = True
        st.rerun()
else:
    edited_system_prompt = st.text_area(
        "Edit system prompt", value=system_prompt, height=300
    )

    col1, col2 = st.columns([1, 4])  # Create columns with ratio 1:4
    with col1:
        col3, col4 = st.columns(
            [1, 1], gap="small"
        )  # Create two equal columns with small gap
        with col3:
            if st.button("Cancel", type="primary", help="Discard changes"):
                st.session_state.editing_system_prompt = False
                st.rerun()
        with col4:
            if st.button("Update"):
                try:
                    os.makedirs(os.path.dirname(system_prompt_path), exist_ok=True)
                    with open(system_prompt_path, "w") as f:
                        f.write(edited_system_prompt)
                    st.success("System prompt saved successfully")
                    st.session_state.editing_system_prompt = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving system prompt: {str(e)}")

# ------------------------------- Document loading section -------------------------------
st.header("Load Documents")
# Initialize session state for files and loaded status if not exists
if "files" not in st.session_state:
    st.session_state.files = {}  # {filename: {content: str, loaded: bool}}
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}  # {filename: content} for caching

# File uploader section
uploaded_files = st.file_uploader(
    "Drop files here", accept_multiple_files=True, key="file_uploader"
)

if uploaded_files:
    # Add new files to session state
    for file in uploaded_files:
        if file.name not in st.session_state.files:
            st.session_state.files[file.name] = {"content": None, "loaded": False}

# Display files and loading status
for filename, file_data in st.session_state.files.items():
    col1, col2, col3 = st.columns([3, 4, 1])

    with col1:
        st.text(filename)

    with col2:
        if file_data["loaded"]:
            token_count = model.count_tokens(file_data["content"])
            st.markdown(f"✅ Loaded ({token_count} tokens)")
        elif "progress" in file_data:
            st.info(f"⏳ Loading... {file_data['progress']:.0%}")
        else:
            st.warning("⏳ Not loaded")

    with col3:
        if st.button("Remove", key=f"remove_{filename}"):
            del st.session_state.files[filename]
            st.rerun()

# Load documents button
if st.session_state.files and st.button("Load Documents"):
    progress_bar = st.progress(0)
    total_files = len(st.session_state.files)

    # Initialize global document content
    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
    
    # Initialize total tokens state if not exists
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    for i, (filename, file_data) in enumerate(st.session_state.files.items()):
        if not file_data["loaded"]:
            try:
                # Check if file is in cache
                if filename in st.session_state.file_cache:
                    file_content = st.session_state.file_cache[filename]
                else:
                    # Get the uploaded file object from st.session_state
                    file_obj = [
                        f for f in st.session_state.file_uploader 
                        if f.name == filename
                    ][0]

                    # Save temporarily to disk or handle in memory
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(filename)[1]
                    ) as tmp_file:
                        tmp_file.write(file_obj.getvalue())
                        tmp_path = tmp_file.name

                    # Convert PDF to images
                    images = convert_from_path(tmp_path, dpi=300)

                    # Open PDF with fitz
                    pdf_document = fitz.open(tmp_path)

                    file_content = ""

                    # Process each page
                    for page_number in range(len(pdf_document)):
                        page = pdf_document.load_page(page_number)
                        if page.get_drawings():  # OCR needed for vector content
                            text = pytesseract.image_to_string(
                                images[page_number], lang="eng"
                            )
                        else:  # Extract text directly
                            text = page.get_text("text")
                        file_content += text + "\n"

                    pdf_document.close()

                    # Clean up temporary file
                    os.unlink(tmp_path)

                    # Store in cache
                    st.session_state.file_cache[filename] = file_content

                # Update session state with file content
                st.session_state.files[filename]["content"] = file_content
                st.session_state.files[filename]["loaded"] = True
                st.session_state.document_content += file_content

                # Update progress
                progress = (i + 1) / total_files
                st.session_state.files[filename]["progress"] = progress
                progress_bar.progress(progress)

            except Exception as e:
                st.error(f"Error loading {filename}: {str(e)}")
                continue

    # Calculate total tokens across all documents and store in session state
    st.session_state.total_tokens = sum(
        model.count_tokens(file_data["content"]).total_tokens
        for file_data in st.session_state.files.values() 
        if file_data["loaded"]
    )
    st.info(f"📄 Total tokens across all documents: {st.session_state.total_tokens:,}")

    # Force a rerun to update the UI
    st.rerun()

# Display total tokens if they exist
if "total_tokens" in st.session_state and st.session_state.total_tokens > 0:
    st.info(f"📄 Total tokens across all documents: {st.session_state.total_tokens:,}")

# ------------------------------- Prompts section -------------------------------
def load_prompt(prompt_num):
    try:
        with open(f"data/{prompt_num}/prompt.txt", "r") as f:
            return f.read()
    except Exception as e:
        return ""


def load_id(prompt_num):
    try:
        with open(f"data/{prompt_num}/id.txt", "r") as f:
            return f.read()
    except Exception as e:
        return ""


def load_expected(prompt_num):
    try:
        with open(f"data/{prompt_num}/expected.txt", "r") as f:
            return f.read()
    except Exception as e:
        return ""


def save_prompt(prompt_num, content):
    try:
        os.makedirs(f"data/{prompt_num}", exist_ok=True)
        with open(f"data/{prompt_num}/prompt.txt", "w") as f:
            f.write(content)
        return True
    except Exception as e:
        return False


def save_expected(prompt_num, content):
    try:
        os.makedirs(f"data/{prompt_num}", exist_ok=True)
        with open(f"data/{prompt_num}/expected.txt", "w") as f:
            f.write(content)
        return True
    except Exception as e:
        return False

st.header("Prompts")

# Add toggle for free tier
use_free_tier = st.toggle("Free Gemini tier (adds timer if running all prompts)", value=False)

# Get list of all folders in data directory
data_folders = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]

# Add run all prompts button
if st.button("▶️▶️▶️ Run all prompts"):
    if "document_content" in st.session_state:
        # Store responses in session state to display them under each prompt
        if "all_responses" not in st.session_state:
            st.session_state.all_responses = {}
        
        for i in range(len(data_folders)):
            prompt = load_prompt(i)
            expected = load_expected(i)
            
            try:
                response = model.generate_content(
                    f"""
                    {system_prompt}

                    ---
                    {st.session_state.document_content}

                    ---

                    {prompt}
                    """,
                    generation_config={"temperature": 0},
                )
                
                # Compare expected and actual results
                comparison_response = model.generate_content(
                    f"""
                    Compare these two texts and return only 'True' if they convey the same meaning,
                    or 'False' if they differ in meaning.
                    Don't worry about units as long as the numerical value are the same.

                    Do not add anything else. Just one word: 'True' or 'False'.
                    
                    Expected:
                    {expected}
                    
                    Actual:
                    {response.text}

                    The meaning of the expected response and the actual response is the same. This statement is: 
                    """,
                    generation_config={
                        "temperature": 0,
                        "candidate_count": 1
                    },
                )
                
                is_accurate = comparison_response.text.strip().lower() == 'true'
                
                # Store response and accuracy for this prompt
                st.session_state.all_responses[i] = {
                    "response": response.text,
                    "is_accurate": is_accurate
                }

                if use_free_tier:
                    time.sleep(60)  # Wait 60 seconds between prompts on free tier

            except Exception as e:
                st.session_state.all_responses[i] = {
                    "error": str(e)
                }
                continue
        st.rerun()  # Rerun to display all responses
    else:
        st.warning("Please load documents first")

# Display prompts section
for i in range(len(data_folders)):
    id = load_id(i)
    st.subheader(f"Prompt {id}")

    # Initialize editing state for this prompt if not present
    if f"editing_prompt_{i}" not in st.session_state:
        st.session_state[f"editing_prompt_{i}"] = False

    prompt = load_prompt(i)

    # Display prompt with edit button or editing interface
    if not st.session_state[f"editing_prompt_{i}"]:
        st.text_area(
            "Prompt", value=prompt, disabled=True, height=150, key=f"display_prompt_{i}"
        )
        if st.button("Edit", key=f"edit_button_{i}"):
            st.session_state[f"editing_prompt_{i}"] = True
            st.rerun()
    else:
        edited_prompt = st.text_area(
            "Edit prompt", value=prompt, height=150, key=f"edit_prompt_{i}"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            col3, col4 = st.columns([1, 1], gap="small")
            with col3:
                if st.button(
                    "Cancel", type="primary", key=f"cancel_{i}", help="Discard changes"
                ):
                    st.session_state[f"editing_prompt_{i}"] = False
                    st.rerun()
            with col4:
                if st.button("Save", key=f"save_{i}"):
                    if save_prompt(i, edited_prompt):
                        st.success("Prompt saved successfully")
                        st.session_state[f"editing_prompt_{i}"] = False
                        st.rerun()
                    else:
                        st.error("Error saving prompt")

    # Expected result
    expected = load_expected(i)
    if not st.session_state.get(f"editing_expected_{i}", False):
        st.text_area(
            "Expected result",
            value=expected,
            disabled=True,
            height=100,
            key=f"display_expected_{i}",
        )
        if st.button("Edit", key=f"edit_expected_button_{i}"):
            st.session_state[f"editing_expected_{i}"] = True
            st.rerun()
    else:
        edited_expected = st.text_area(
            "Edit expected result", value=expected, height=100, key=f"edit_expected_{i}"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            col3, col4 = st.columns([1, 1], gap="small")
            with col3:
                if st.button(
                    "Cancel",
                    type="primary",
                    key=f"cancel_expected_{i}",
                    help="Discard changes",
                ):
                    st.session_state[f"editing_expected_{i}"] = False
                    st.rerun()
            with col4:
                if st.button("Save", key=f"save_expected_{i}"):
                    if save_expected(i, edited_expected):
                        st.success("Expected result saved successfully")
                        st.session_state[f"editing_expected_{i}"] = False
                        st.rerun()
                    else:
                        st.error("Error saving expected result")

    # Run prompt button
    if st.button("▶️ Run Prompt", key=f"run_{i}"):
        if "document_content" in st.session_state:
            try:
                response = model.generate_content(
                    f"""
                    {system_prompt}

                    ---
                    {st.session_state.document_content}

                    ---

                    {prompt}
                    """,
                    generation_config={"temperature": 0},
                )
                st.write("Response:")
                st.write(response.text)

                # Compare expected and actual results
                comparison_response = model.generate_content(
                    f"""
                    Compare these two texts and return only 'True' if they convey the same meaning,
                    or 'False' if they differ in meaning.
                    Don't worry about units as long as the numerical value are the same.

                    Do not add anything else. Just one word: 'True' or 'False'.
                    
                    Expected:
                    {expected}
                    
                    Actual:
                    {response.text}

                    The meaning of the expected response and the actual response is the same. This statement is: 
                    """,
                    generation_config={
                        "temperature": 0,
                        "candidate_count": 1
                    },
                )
                
                is_accurate = comparison_response.text.strip().lower() == 'true'
                
                if is_accurate:
                    st.markdown(
                        """
                        <div style='background-color: #90EE90; padding: 10px; border-radius: 5px;'>
                            ✅ Response matches expected result
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style='background-color: #FFB6C1; padding: 10px; border-radius: 5px;'>
                            ❌ Response differs from expected result
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error running prompt: {str(e)}")
        else:
            st.warning("Please load documents first")

    # After the prompt and expected result displays, show the response if it exists
    if "all_responses" in st.session_state and i in st.session_state.all_responses:
        response_data = st.session_state.all_responses[i]
        
        if "error" in response_data:
            st.error(f"Error running prompt: {response_data['error']}")
        else:
            st.write("Response:")
            st.write(response_data["response"])
            
            if response_data["is_accurate"]:
                st.markdown(
                    """
                    <div style='background-color: #90EE90; padding: 10px; border-radius: 5px;'>
                        ✅ Response matches expected result
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style='background-color: #FFB6C1; padding: 10px; border-radius: 5px;'>
                        ❌ Response differs from expected result
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()
