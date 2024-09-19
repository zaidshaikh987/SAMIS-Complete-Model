from googletrans import Translator
import json

class LocalizedSupport:
    def __init__(self, target_language='hi'):
        self.translator = Translator()
        self.target_language = target_language

    def translate_text(self, text):
        try:
            translated = self.translator.translate(text, dest=self.target_language)
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def simplify_insights(self, insights):
        # Example simplification logic
        simple_insights = []
        for insight in insights:
            if "increase" in insight.lower():
                simple_insights.append("The price is likely to go up.")
            elif "decrease" in insight.lower():
                simple_insights.append("The price is likely to go down.")
            else:
                simple_insights.append("The price trend is unclear.")
        return simple_insights

    def create_chatbot_response(self, user_input):
        responses = {
            "price": "The current price prediction is...",
            "weather": "The current weather forecast is...",
            "recommendation": "Based on the data, we recommend..."
        }
        return responses.get(user_input.lower(), "Sorry, I didn't understand that.")

def main():
    # Sample data
    model_output = ["Prices are expected to increase due to high demand.", 
                    "Weather conditions are favorable for the crop yield."]
    
    # Initialize LocalizedSupport
    support = LocalizedSupport(target_language='hi')  # Set target language to Hindi

    # Translate model output
    translated_output = [support.translate_text(text) for text in model_output]
    
    # Simplify insights
    simple_insights = support.simplify_insights(model_output)
    
    # Print results
    print("Translated Output:", translated_output)
    print("Simplified Insights:", simple_insights)
    
    # Simulate chatbot interaction
    user_input = "price"
    response = support.create_chatbot_response(user_input)
    print("Chatbot Response:", response)

if __name__ == "__main__":
    main()
