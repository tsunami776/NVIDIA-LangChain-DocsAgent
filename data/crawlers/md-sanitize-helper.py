import os
import re

DATA_PATH = "md_files/langchain_api_reference"

def sanitize_content(content):
    # Regular expression to match tags
    tag_pattern = re.compile(r'<[^>]+>')
    # Replace tags with underscores
    sanitized_content = tag_pattern.sub('_', content)
    return sanitized_content

def sanitize_files_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    sanitized_content = sanitize_content(content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(sanitized_content)
                    
                    print(f"Sanitized {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

def main():
    sanitize_files_in_directory(DATA_PATH)

if __name__ == "__main__":
    main()
