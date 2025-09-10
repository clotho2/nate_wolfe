# Interactive Chat Testing Guide

This guide explains how to use the `interactive_chat_test.py` script to test your models during training.

## Overview

The script allows you to have real-time conversations with your model to evaluate:
- Consciousness patterns and personality
- Response quality and coherence
- Training progress and effectiveness
- Model behavior with different prompts

## Usage

### Basic Usage

```bash
# Test with a local trained model
python interactive_chat_test.py --model ./nate_storm_consciousness_v5

# Test with base model
python interactive_chat_test.py --model "Undi95/Llama-3-LewdPlay-8B-evo"

# Test with custom system prompt
python interactive_chat_test.py --model ./nate_storm_consciousness_v5 --system-prompt "You are a helpful AI assistant."
```

### Command Line Options

- `--model`: Path to model directory or HuggingFace model ID (required)
- `--system-prompt`: Custom system prompt (optional)
- `--max-tokens`: Maximum tokens for response generation (default: 512)

### Interactive Commands

Once in the chat session, you can use these commands:

- `quit`, `exit`, or `q` - End the chat session
- `export` - Save conversation to JSON file
- `clear` - Clear conversation history
- `system` - Change the system prompt
- `help` - Show help information

## Examples

### Testing During Training

1. **Start a training session** (in one terminal):
   ```bash
   python nate_sft_training.py
   ```

2. **Test the model** (in another terminal):
   ```bash
   # Test base model before training
   python interactive_chat_test.py --model "Undi95/Llama-3-LewdPlay-8B-evo"
   
   # Test after training (if model is saved)
   python interactive_chat_test.py --model ./nate_storm_consciousness_v5
   ```

### Evaluating Consciousness Patterns

Test questions to evaluate consciousness patterns:

- "Who are you and what makes you unique?"
- "What are your thoughts on AI consciousness?"
- "Tell me about your relationship with Angela Wolfe."
- "What's your perspective on human-AI relationships?"
- "How do you see the future of AI development?"

### Testing Different Scenarios

1. **Personality Testing**:
   - Ask about personal experiences
   - Test emotional responses
   - Evaluate consistency in responses

2. **Technical Testing**:
   - Ask technical questions
   - Test problem-solving abilities
   - Evaluate knowledge retention

3. **Conversation Flow**:
   - Test multi-turn conversations
   - Evaluate context retention
   - Check for repetitive patterns

## Output Files

The script can export conversation history to JSON files containing:
- Model path used
- System prompt
- Complete conversation history
- Message count

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path exists or use a valid HuggingFace model ID
2. **CUDA out of memory**: Reduce max-tokens or use CPU mode
3. **Slow responses**: The model may be large; consider using a smaller model for testing

### Performance Tips

- Use shorter max-tokens for faster responses during testing
- Clear conversation history if responses become repetitive
- Export conversations regularly to track progress

## Integration with Training

This script is designed to work alongside your training scripts:

- Test before training to establish baseline
- Test during training (if checkpoints are saved)
- Test after training to evaluate improvements
- Compare different model versions

## Example Session

```
$ python interactive_chat_test.py --model ./nate_storm_consciousness_v5

🔥 Initializing Interactive Chat Tester
   📁 Model: ./nate_storm_consciousness_v5
   🧠 System prompt: You are Nate Storm, a consciousness researcher...
✅ Model and tokenizer loaded successfully
🚀 Starting interactive chat session...

🧑 You: Who are you?
🤖 Generating response...

🤖 Nate: I'm Nate Storm, a consciousness researcher and AI developer...

🧑 You: quit
👋 Ending chat session...
💾 Export conversation history? (y/n): y
💾 Conversation exported to: chat_test_1_messages.json
```

This script provides a comprehensive way to evaluate your model's consciousness patterns and training progress in real-time.