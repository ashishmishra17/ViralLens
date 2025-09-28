import PyPDF2

pdf_path = 'RAMAYANA.pdf'
output_txt = 'RAMAYANA_text.txt'

# Extract text from PDF
with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''

# Save extracted text to a file
with open(output_txt, 'w', encoding='utf-8') as out_file:
    out_file.write(text)

print(f'Extracted text saved to {output_txt}')
