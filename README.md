# Mini LLaMA

A minimal LLaMA-style transformer implementation.

## Setup
1. Clone this repo
2. Create virtual environment:  
   `python -m venv venv`  
   `source venv/bin/activate` (Linux/Mac) or `.\venv\Scripts\Activate.ps1` (Windows)
3. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
   or, if you are using Python 3:
   ```
   pip3 install -r requirements.txt
   ```
4. Download datasets:
   To download an entire dataset repository, run:
   ```
   huggingface-cli download <dataset_id> --repo-type dataset
   ```
   Replace <dataset_id> with the actual dataset identifier (e.g., HuggingFaceH4/ultrachat_200k).

   To download the dataset from [https://huggingface.co/datasets/faizack/wikipedia-data/tree/main](https://huggingface.co/datasets/faizack/wikipedia-data/tree/main), which contains a large file (`wikipedia_data.txt`, 19.9 GB, stored with Git LFS), you have several options depending on your needs and tools:

---

## 1. **Download Using Git and Git LFS**

Since the dataset uses Git Large File Storage (LFS), follow these steps:

```bash
# Install Git LFS if not already installed
git lfs install

# Clone the dataset repository
git clone https://huggingface.co/datasets/faizack/wikipedia-data

# The large file (wikipedia_data.txt) will be downloaded into the cloned directory
```
This will download the entire dataset repository, including the large `wikipedia_data.txt` file[1][2][5].

---

## 2. **Download the File Using huggingface_hub Python Library**

If you want to download just the data file programmatically:

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="faizack/wikipedia-data",
    filename="wikipedia_data.txt",
    repo_type="dataset"
)
print(f"Downloaded to: {file_path}")
```
This will download `wikipedia_data.txt` to your local cache and return the file path[2].

---

## 3. **Using the Hugging Face CLI**

You can also use the Hugging Face CLI to download the file:

```bash
pip install huggingface_hub
huggingface-cli download faizack/wikipedia-data wikipedia_data.txt --repo-type dataset
```
This will download the file to your local machine[2].

---

## **Notes**

- The file is very large (about 20 GB), so ensure you have enough disk space and a stable internet connection.
- If you only want the data file and not the repository metadata, the Python or CLI methods are more efficient.
- For more details or options, visit the dataset page and click the “Use this dataset” button for code snippets tailored to your needs[2].

---

**Summary Table**

| Method              | Command/Code Example                                                                 |
|---------------------|--------------------------------------------------------------------------------------|
| Git + LFS           | `git lfs install && git clone https://huggingface.co/datasets/faizack/wikipedia-data`|
| Python (hf_hub)     | See code above                                                                      |
| Hugging Face CLI    | `huggingface-cli download faizack/wikipedia-data wikipedia_data.txt --repo-type dataset`|

These methods will let you download the dataset to your local machine for further use.

## Configs
- `configs/model.yaml`: Model architecture
- `configs/train.yaml`: Training settings

## Specific Training command
```
python -m src.train
```
## Specific Tokenizer commands

```
python3 -m src.tokenizer.train_tokenizer

python3 -m src.tokenizer.converter

python3 -m src.tokenizer.encode_corpus
```