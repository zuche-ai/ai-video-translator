# LDS Glossary System

## Overview

The LDS Glossary System enhances translation accuracy for Latter-day Saint (LDS) content by providing contextually appropriate translations for LDS-specific terms and phrases. This system automatically applies these specialized translations after the general translation process, ensuring that LDS terminology is accurately conveyed in the target language.

## Features

- **CSV-based Storage**: Easy-to-edit CSV file format for term management
- **Automatic Integration**: Seamlessly integrated with the translation pipeline
- **Context-Aware**: Preserves context and meaning of LDS terms
- **Community-Driven**: Easy to update and contribute new terms
- **Case-Insensitive Matching**: Handles variations in capitalization
- **Word Boundary Protection**: Prevents partial word replacements

## File Structure

```
video_translator/
├── lds_glossary.csv              # Main glossary file
├── video_translator/core/
│   ├── glossary_loader.py        # Glossary loading and management
│   └── translator.py             # Updated with glossary integration
└── test_lds_glossary.py         # Test suite
```

## CSV Format

The `lds_glossary.csv` file contains three columns:

| Column | Description | Example |
|--------|-------------|---------|
| `original_term` | English LDS term | "temple" |
| `translated_term` | Target language translation | "templo" |
| `context_notes` | Brief explanation of the term | "Place of worship and ordinances" |

### Example Entry
```csv
original_term,translated_term,context_notes
temple,templo,Place of worship and ordinances
bishop,obispo,Local congregation leader
testimony,testimonio,Personal witness of truth
```

## Usage

### Automatic Usage

The glossary is automatically applied when translating to Spanish (`es`). No additional configuration is required.

```python
from video_translator.core.translator import translate_segments

segments = [{"text": "I love going to the temple.", "start": 0, "end": 5}]
translated = translate_segments(segments, "en", "es")
# Result: "Me encanta ir al templo." (with "templo" correctly translated)
```

### Manual Usage

You can also apply the glossary manually to any text:

```python
from video_translator.core.glossary_loader import apply_lds_glossary

text = "The bishop gave a great talk about testimony."
processed = apply_lds_glossary(text)
# Result: "The obispo gave a great talk about testimonio."
```

### Programmatic Management

```python
from video_translator.core.glossary_loader import LDSGlossaryLoader

# Load the glossary
loader = LDSGlossaryLoader()

# Add a new term
loader.add_term("stake conference", "conferencia de estaca", "Stake-level meeting")

# Get statistics
stats = loader.get_glossary_stats()
print(f"Total terms: {stats['total_terms']}")

# List all terms
terms = loader.list_terms()
for term in terms:
    print(f"{term['original']} -> {term['translated']}")

# Save changes back to CSV
loader.save_glossary()
```

## Current Terms

The glossary currently includes 75+ LDS-specific terms covering:

### Church Organization
- temple, ward, stake, bishop, stake president
- general conference, relief society, priesthood
- primary, young men, young women, seminary, institute

### Religious Concepts
- testimony, covenant, ordinance, endowment, sealing
- blessing, atonement, repentance, baptism, confirmation
- holy ghost, heavenly father, savior, redeemer

### Church Programs
- ministering, family home evening, family history
- temple work, missionary work, fast and testimony meeting
- sacrament meeting, home teaching

### Doctrinal Terms
- gospel, plan of salvation, eternal life, exaltation
- celestial kingdom, terrestrial kingdom, telestial kingdom
- grace, faith, hope, charity, virtue, knowledge

### Church Practices
- tithing, fast offering, word of wisdom, sabbath day
- temple recommend, worthiness, chastity
- patriarchal blessing, obedience, service

## Adding New Terms

### Method 1: Edit CSV Directly

1. Open `lds_glossary.csv` in a text editor or spreadsheet application
2. Add a new row with the format: `original_term,translated_term,context_notes`
3. Save the file

### Method 2: Programmatic Addition

```python
from video_translator.core.glossary_loader import LDSGlossaryLoader

loader = LDSGlossaryLoader()
loader.add_term("new term", "nuevo término", "Description of the term")
loader.save_glossary()
```

### Guidelines for Adding Terms

1. **Accuracy**: Ensure translations are contextually appropriate for LDS usage
2. **Consistency**: Use consistent terminology across the glossary
3. **Completeness**: Include both singular and plural forms if they differ significantly
4. **Context**: Provide helpful context notes for future reference
5. **Validation**: Test new terms with the test suite

## Testing

Run the comprehensive test suite:

```bash
python test_lds_glossary.py
```

The test suite verifies:
- Glossary loading and statistics
- Individual term translations
- Text replacement functionality
- Translator integration
- Glossary management features

## Community Contributions

### Contributing New Terms

1. **Fork the repository** (if applicable)
2. **Add terms** to `lds_glossary.csv`
3. **Test your changes** with `python test_lds_glossary.py`
4. **Submit a pull request** with a description of the added terms

### Quality Assurance

Before contributing:
- Verify translations with native speakers
- Ensure terms are commonly used in LDS contexts
- Test with real LDS content
- Follow the existing format and style

## Technical Details

### Implementation

The glossary system uses:
- **Regular Expressions**: For precise word boundary matching
- **Case-Insensitive Matching**: To handle capitalization variations
- **Longest-First Replacement**: To avoid partial word conflicts
- **CSV Parsing**: For easy maintenance and updates

### Performance

- **Fast Loading**: Glossary loads once and caches in memory
- **Efficient Matching**: Uses compiled regex patterns
- **Minimal Overhead**: Adds negligible time to translation process

### Error Handling

- **Graceful Degradation**: System continues working if glossary file is missing
- **Logging**: Comprehensive logging for debugging
- **Validation**: Input validation and error reporting

## Future Enhancements

### Planned Features

1. **Multi-Language Support**: Extend beyond Spanish to other languages
2. **Context-Aware Matching**: Consider surrounding words for better accuracy
3. **Machine Learning**: Learn from corrections and improve over time
4. **Web Interface**: Web-based glossary management tool
5. **Version Control**: Track changes and allow rollbacks

### Potential Improvements

- **Fuzzy Matching**: Handle typos and variations
- **Phrase Matching**: Support multi-word phrases
- **Regional Variations**: Handle different Spanish dialects
- **Usage Statistics**: Track which terms are used most frequently

## Troubleshooting

### Common Issues

1. **Terms Not Replaced**: Check spelling and case sensitivity
2. **Partial Replacements**: Ensure word boundaries are respected
3. **Missing Glossary**: Verify `lds_glossary.csv` exists and is readable
4. **Encoding Issues**: Ensure CSV file uses UTF-8 encoding

### Debug Mode

Enable debug output to see what's happening:

```python
from video_translator.core.translator import translate_segments

segments = [{"text": "test text", "start": 0, "end": 5}]
translated = translate_segments(segments, "en", "es", debug=True)
```

## Support

For questions, issues, or contributions:
1. Check the test suite for examples
2. Review the CSV file format
3. Test with the provided test script
4. Consult the code documentation

---

*This glossary system represents a significant step forward in accurately translating LDS content while preserving the spiritual and cultural context that makes these translations meaningful.* 