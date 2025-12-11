import os 
import hashlib 
from dotenv import load_dotenv 

load_dotenv()

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file to detect duplicates.
    Same file = Same hash, Different file = Different hash
    """
    sha256_has = hashlib.sha256()

    with open(file_path,'rb') as f:
        # Read file in chunks (1MB at a time to handle large PDFs)
        for byte_block in iter(lambda: f.read(1024*1024),b""):
            sha256_has.update(byte_block)

    return sha256_has.hexdigest()



if __name__ == '__main__':

    pdf_path = "sample.pdf"

    if os.path.exists(pdf_path):
        hash_value = calculate_file_hash(pdf_path)
        print(f"\n📄 File: {pdf_path}")
        print(f"🔑 Hash: {hash_value}")
        
        # Calculate again to prove same file = same hash
        hash_value_2 = calculate_file_hash(pdf_path)
        print(f"\n🔁 Calculated again: {hash_value_2}")
        print(f"✅ Hashes match: {hash_value == hash_value_2}")
    else:
        print(f"\n❌ Please add '{pdf_path}' to root folder to test")
