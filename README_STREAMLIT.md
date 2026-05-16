# Project Scheduler - Streamlit App

A modern web application for AI-powered project scheduling with task breakdown, dependency management, and Gantt chart visualization.

## Features

✨ **AI-Powered Task Breakdown** - Uses CrewAI with GPT-4 to intelligently break down projects  
📊 **Interactive Dashboard** - Clean, modern UI with multiple views  
📈 **Gantt Charts** - Visual timeline representation of tasks and dependencies  
📋 **Detailed Reports** - Generate Word documents with project specifications  
💾 **Export Options** - Download tasks, schedules, and charts  
⚡ **Real-Time Processing** - Watch your project get structured in real-time  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API key from [OpenAI](https://platform.openai.com/api-keys)

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Step 1: Input Tab
- Describe your project in detail
- Include scope, timeline, team size, budget, and key requirements
- Click "Generate Schedule" to process

### Step 2: Tasks Tab
- View all generated tasks with details
- See duration, cost, resources, and dependencies
- Download tasks as CSV

### Step 3: Schedule Tab
- View the complete project timeline
- See the Gantt chart visualization
- Download the schedule as CSV

### Step 4: Report Tab
- Download the Word document report
- Download the Gantt chart image
- View project summary metrics

## Project Structure

```
Tipsy/
├── app.py                    # Streamlit application
├── main.py                   # FastAPI backend (optional)
├── requirements.txt          # Python dependencies
├── agents/
│   └── crew.py              # CrewAI configuration
├── core/
│   ├── models.py            # Data models
│   ├── scheduler.py         # Task scheduling logic
│   └── validator.py         # Project validation
├── tools/
│   ├── doc_generator.py     # Word document generation
│   └── gantt_generator.py   # Gantt chart generation
└── utils/
    └── parser.py            # JSON parsing utilities
```

## UI/UX Features

### Clean Design
- Modern, intuitive interface
- Color-coded sections and alerts
- Responsive layout for all screen sizes

### User Guidance
- Helpful tips and examples
- Input placeholders with suggestions
- Status spinners during processing

### Data Presentation
- Summary metrics and KPIs
- Interactive tables
- Visual charts and diagrams

### Accessibility
- Large, readable text
- Clear navigation with tabs
- Descriptive labels and help text

## Tips for Best Results

1. **Be Specific** - Include as much detail as possible in your project description
2. **Include Context** - Mention team size, budget, and timeline constraints
3. **Mention Milestones** - Key deliverables help the AI understand priorities
4. **Review and Adjust** - The AI provides a baseline; adjust as needed

## Example Project Description

```
Build a mobile app for e-commerce with user authentication, product catalog, 
shopping cart, payment processing, and order tracking. Timeline: 3 months, 
Team: 5 people (1 PM, 2 backend devs, 1 frontend dev, 1 QA), Budget: $50,000. 
Key milestones: MVP in 6 weeks, beta testing in 8 weeks, launch in 12 weeks.
```

## Troubleshooting

### "Error: OPENAI_API_KEY not found"
- Make sure your `.env` file exists and contains a valid API key
- The key should be from [OpenAI API Keys](https://platform.openai.com/api-keys)

### "Error: safe_parse_json failed"
- The AI response might be malformed. Try a simpler, more specific project description
- Check that your OpenAI API key has sufficient credits

### Generated files not appearing
- Make sure you have write permissions in the project directory
- Check that matplotlib and python-docx are properly installed

## Performance Notes

- First request may take 30-60 seconds as CrewAI processes the project
- Subsequent requests within the same session are cached
- For large projects, consider breaking them into phases

## System Requirements

- Python 3.8 or higher
- 100MB free disk space
- Internet connection for OpenAI API

## License

This project uses CrewAI, Streamlit, and other open-source libraries.

## Support

For issues or questions:
1. Check the sidebar tips in the app
2. Review example project descriptions
3. Verify your OpenAI API key and credits
