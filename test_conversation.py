import unittest
from app import get_ai_response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TestConversationFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the fine-tuned model if available, otherwise use base BLOOM model
        model_path = "./trained_model"
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls.model = AutoModelForCausalLM.from_pretrained(model_path)
        except:
            print("Using base BLOOM model as trained model not found")
            cls.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
            cls.model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

    def test_basic_response(self):
        """Test if the model can generate a basic response"""
        message = "Hello, how are you?"
        response = get_ai_response(message)
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_personality_response(self):
        """Test if the model responds differently with different personalities"""
        message = "What do you think about art?"
        personality1 = "I am an optimistic and creative person who loves art and expresses emotions freely."
        personality2 = "I am a logical and analytical person who prefers science and rational thinking."
        personality3 = "I am a philosophical person who sees deeper meaning in everything."

        response1 = get_ai_response(message, personality1)
        response2 = get_ai_response(message, personality2)
        response3 = get_ai_response(message, personality3)

        # Test distinct responses for different personalities
        self.assertNotEqual(response1, response2)
        self.assertNotEqual(response2, response3)
        self.assertNotEqual(response1, response3)

        # Test response alignment with personality traits
        self.assertTrue(
            any(word in response1.lower() for word in ['creative', 'beautiful', 'emotion', 'love']),
            "Response doesn't reflect optimistic personality"
        )
        self.assertTrue(
            any(word in response2.lower() for word in ['analyze', 'rational', 'perspective', 'think']),
            "Response doesn't reflect analytical personality"
        )

    def test_context_maintenance(self):
        """Test if the model maintains context across complex conversations"""
        context = ""
        personality = "I am a friendly and attentive assistant who remembers details."
        messages = [
            "My name is Alice and I love painting landscapes.",
            "What's my favorite art form?",
            "Do you remember my name and what I like to paint?"
        ]

        # Build context through conversation
        for message in messages:
            response = get_ai_response(message, personality, context)
            context += f"User: {message}\nAssistant: {response}\n"

        # Check if the model remembers multiple context elements
        self.assertTrue(
            'Alice' in context.lower() and 'landscape' in context.lower(),
            "Model failed to maintain multiple context elements"
        )

        # Test context with emotional state
        emotional_context = "User seems excited about their recent painting."
        response = get_ai_response("How do you feel about my progress?", personality, emotional_context)
        self.assertTrue(
            any(word in response.lower() for word in ['excited', 'progress', 'wonderful', 'great']),
            "Model failed to respond appropriately to emotional context"
        )

    def test_response_length(self):
        """Test if the model generates responses within reasonable length"""
        message = "Tell me a short story."
        response = get_ai_response(message)
        
        # Response should be between 10 and 500 characters
        self.assertTrue(10 <= len(response) <= 500)

    def test_creative_response(self):
        """Test if the model can generate creative responses"""
        creative_prompt = "Write a unique metaphor about life."
        response = get_ai_response(creative_prompt)

        # Response should be unique and contain figurative language
        self.assertTrue(
            len(response) > 0,
            "Model failed to generate creative response"
        )

def evaluate_model_performance():
    """Run all tests and print a summary"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConversationFlow)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    print("\nModel Evaluation Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")

if __name__ == '__main__':
    evaluate_model_performance()