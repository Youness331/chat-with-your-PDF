# AI PDF Chat Assistant 🤖

A Streamlit application that allows you to chat with your PDF documents using Google's Gemini AI.

## Features

- 📄 Upload and process PDF documents
- 💬 Chat interface with your PDF content
- 🤖 Powered by Google Gemini 1.5 Flash
- 🎨 Dark theme UI
- 📚 Chat history in sidebar
- ⚡ Fast vector-based document retrieval

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Youness331/chat-with-your-PDF.git
cd QA-Bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
1. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Run the application
```bash
streamlit run app_final.py
```

## Usage

1. **Upload PDF**: Click "Upload your PDF" and select a PDF file
2. **Process**: Click "Process PDF" to analyze the document
3. **Chat**: Use the form at the bottom to ask questions about your PDF

## File Structure

```
QA Bot/
├── app_final.py          # Main application (recommended)
├── app_clean.py          # Alternative clean version
├── app_simple.py         # Simplified version
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Dependencies

- streamlit
- google-generativeai
- langchain
- langchain-community
- chromadb
- sentence-transformers
- pypdf
- python-dotenv

## Security Note

⚠️ **Never commit your `.env` file or API keys to GitHub!** 

The `.gitignore` file is configured to exclude sensitive files.

## Troubleshooting

### API Key Issues
- Make sure your `.env` file is in the project root
- Verify your Gemini API key is valid
- Check that the `.env` file is not committed to git

### PDF Processing Issues
- Ensure your PDF contains text (not just images)
- Try with a different PDF if processing fails
- Check file size (very large PDFs may take longer)

### 403 Errors
- This usually indicates API key issues
- Verify your Gemini API key is correct and active
- Check if you've exceeded API quotas

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License