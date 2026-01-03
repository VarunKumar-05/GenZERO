import markdown
import os

# Configuration
INPUT_FILE = r"d:\Hackathons  & Competitions\Synaptix\Model-Speech\TECHNICAL_DOC.md"
OUTPUT_FILE = r"d:\Hackathons  & Competitions\Synaptix\Model-Speech\TECHNICAL_DOC.html"

# CSS for better printing/viewing (GitHub-like style)
CSS = """
<style>
    body {
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
        line-height: 1.6;
        color: #24292e;
        max-width: 900px;
        margin: 0 auto;
        padding: 40px;
    }
    h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; }
    code { background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }
    pre { background-color: #f6f8fa; padding: 16px; overflow: auto; border-radius: 3px; }
    blockquote { border-left: .25em solid #dfe2e5; color: #6a737d; padding: 0 1em; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
    th, td { border: 1px solid #dfe2e5; padding: 6px 13px; }
    th { background-color: #f6f8fa; }
    img { max-width: 100%; }
    
    @media print {
        body { max-width: 100%; padding: 0; }
        a { text-decoration: none; color: black; }
    }
</style>
"""

def convert():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Convert to HTML
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # Wrap in full HTML document
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Technical Documentation</title>
        {CSS}
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    print(f"Successfully created {OUTPUT_FILE}")

if __name__ == "__main__":
    convert()
